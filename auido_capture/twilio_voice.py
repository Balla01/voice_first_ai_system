"""
twilio_voice.py — Twilio phone integration (telephony front door for Layer 1/2).

Two call directions, both converging on the same /voice webhook's TwiML:
  Inbound:  customer dials TWILIO_PHONE_NUMBER -> Twilio POSTs to /twilio/voice
  Outbound: POST /twilio/call {"to": "+1..."}   -> Twilio dials the customer,
            and once answered, POSTs to /twilio/voice (reused as the
            outbound call's `url`)

/twilio/voice's TwiML does two things concurrently:
  1. <Start><Stream> forks a copy of the customer's audio (inbound_track
     only — the agent's leg isn't part of this call) to /ws/twilio in
     main.py, tagged with session_id=CallSid.
  2. <Dial> bridges the call to the human agent's real phone number, so the
     agent talks to the customer exactly as before — just over Twilio
     instead of whatever softphone/system-audio setup was used previously.

The agent joins the SAME session on the frontend (capture-client.js's
joinCall(sessionId)) using the CallSid logged/returned here, so their mic
("agent") and the forked phone audio ("customer") land in one
AudioRouter/session, matching the existing mic+system merge pipeline.

SECURITY NOTE: /twilio/voice verifies Twilio's request signature (only
Twilio can trigger it). /twilio/call has NO auth of its own — it lets
whoever can reach it place a real phone call on your Twilio account's
balance. It's fine while PUBLIC_BASE_URL only points at a local ngrok
tunnel you're not sharing, but before deploying anywhere reachable, put
this behind the same auth your frontend/ops tooling uses.
"""

import logging
import os

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response
from twilio.request_validator import RequestValidator
from twilio.rest import Client
from twilio.twiml.voice_response import Dial, Start, VoiceResponse

logger = logging.getLogger("insureassist.layer1")

router = APIRouter(prefix="/twilio", tags=["twilio"])

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
AGENT_PHONE_NUMBER = os.getenv("AGENT_PHONE_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5500").rstrip("/")

_validator = RequestValidator(TWILIO_AUTH_TOKEN) if TWILIO_AUTH_TOKEN else None


async def _verify_twilio_signature(request: Request) -> None:
    """
    Rejects any /twilio/voice POST that isn't actually signed by Twilio
    (X-Twilio-Signature, see https://www.twilio.com/docs/usage/security).
    This endpoint is public (Twilio must reach it over the internet), so
    without this check anyone could POST forged call events at it.
    """
    if _validator is None:
        raise HTTPException(500, "TWILIO_AUTH_TOKEN is not set — see .env.example")

    signature = request.headers.get("X-Twilio-Signature", "")
    form = await request.form()
    if not _validator.validate(str(request.url), dict(form), signature):
        logger.warning(f"Rejected /twilio/voice webhook with invalid signature from {request.client}")
        raise HTTPException(403, "Invalid Twilio signature")


def _ws_base_url() -> str:
    if not PUBLIC_BASE_URL:
        raise HTTPException(500, "PUBLIC_BASE_URL is not set — see .env.example")
    return PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")


@router.post("/voice")
async def voice_webhook(request: Request, CallSid: str = Form(...)):
    """
    Twilio's Voice webhook — used for BOTH inbound calls (set as the Twilio
    number's "A call comes in" webhook) and outbound calls (passed as the
    `url` in the REST call /twilio/call creates below). Either way, CallSid
    is unique per call and doubles as the session_id the agent joins on the
    frontend.
    """
    await _verify_twilio_signature(request)

    session_id = CallSid
    join_url = f"{FRONTEND_URL}/?session_id={session_id}"
    logger.info(f"Twilio call {session_id}: agent join URL -> {join_url}")

    vr = VoiceResponse()

    start = Start()
    start.stream(url=f"{_ws_base_url()}/ws/twilio?session_id={session_id}", track="inbound_track")
    vr.append(start)

    if not AGENT_PHONE_NUMBER:
        logger.error(f"[{session_id}] AGENT_PHONE_NUMBER is not set — call will not be bridged to an agent")
    else:
        dial = Dial()
        dial.number(AGENT_PHONE_NUMBER)
        vr.append(dial)

    return Response(content=str(vr), media_type="application/xml")


@router.post("/call")
async def place_outbound_call(to: str = Form(...)):
    """
    Triggers an outbound call to a customer: POST /twilio/call with form
    field to=+15551234567. Twilio dials `to`; once answered, it POSTs to
    /twilio/voice (this file's other endpoint), which forks audio and
    bridges to the agent — same TwiML path as an inbound call.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        raise HTTPException(500, "TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_PHONE_NUMBER must be set")
    if not PUBLIC_BASE_URL:
        raise HTTPException(500, "PUBLIC_BASE_URL is not set — see .env.example")

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(to=to, from_=TWILIO_PHONE_NUMBER, url=f"{PUBLIC_BASE_URL}/twilio/voice")

    logger.info(f"Outbound call placed: CallSid={call.sid} to={to}")
    return {"call_sid": call.sid, "session_id": call.sid, "join_url": f"{FRONTEND_URL}/?session_id={call.sid}"}
