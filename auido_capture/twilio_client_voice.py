"""
twilio_client_voice.py — Twilio Voice SDK (browser softphone) integration.

Additive, parallel to twilio_voice.py's phone-bridge flow (/twilio/voice,
/twilio/call, AGENT_PHONE_NUMBER <Dial><Number>). This file does not modify
that flow or anything in main.py's existing routes — it mounts its own
router (see main.py's two added lines) and only reuses twilio_voice.py's
signature-verification/URL helpers by import, never by editing that file.

It replaces the "dial a second real phone" leg with a WebRTC "dial a
browser" leg (<Dial><Client>), so the agent side of a call never needs a
Twilio-verified phone number — only a short-lived Access Token.

Flow:
  Agent's browser -> GET /twilio/token?identity=<agent_id> -> registers a
    Twilio.Device (see frontend/twilio-client.js) under that identity.
  Agent clicks "Answer via Browser" -> POST /twilio/call-client {to, identity}
    -> Twilio dials the customer's phone (same REST call shape as
    twilio_voice.py's place_outbound_call).
  Customer answers -> Twilio POSTs to /voice-client (this file's webhook,
    passed as `url` above) -> TwiML forks customer audio to the EXISTING
    /ws/twilio (untouched, in main.py) and dials <Client>{identity}</Client>
    -> the agent's registered Device gets an incoming call and auto-accepts,
    bridging WebRTC audio directly — no second PSTN leg, so no second
    verified number is ever needed.
"""

import logging
import os

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import Response
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant
from twilio.rest import Client
from twilio.twiml.voice_response import Dial, Start, VoiceResponse

from twilio_voice import (
    _verify_twilio_signature,
    _ws_base_url,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    PUBLIC_BASE_URL,
)

logger = logging.getLogger("insureassist.layer1")

router = APIRouter(prefix="/twilio", tags=["twilio-client"])

TWILIO_API_KEY_SID = os.getenv("TWILIO_API_KEY_SID", "")
TWILIO_API_KEY_SECRET = os.getenv("TWILIO_API_KEY_SECRET", "")
TWILIO_TWIML_APP_SID = os.getenv("TWILIO_TWIML_APP_SID", "")

_DEFAULT_IDENTITY = "agent"


@router.get("/token")
async def issue_access_token(identity: str = Query(default=_DEFAULT_IDENTITY)):
    """
    Mints a short-lived Twilio Access Token (Voice grant) for the agent's
    browser to register a Twilio.Device under `identity` — see
    frontend/twilio-client.js. Independent credential pair
    (TWILIO_API_KEY_SID/SECRET) from TWILIO_AUTH_TOKEN; only used by this
    new browser-softphone path, never by twilio_voice.py's phone-bridge flow.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_API_KEY_SID and TWILIO_API_KEY_SECRET and TWILIO_TWIML_APP_SID):
        raise HTTPException(
            500,
            "TWILIO_API_KEY_SID/TWILIO_API_KEY_SECRET/TWILIO_TWIML_APP_SID must be set "
            "(Console -> API keys & tokens, and Console -> Voice -> TwiML Apps) — see .env",
        )

    voice_grant = VoiceGrant(outgoing_application_sid=TWILIO_TWIML_APP_SID, incoming_allow=True)
    token = AccessToken(TWILIO_ACCOUNT_SID, TWILIO_API_KEY_SID, TWILIO_API_KEY_SECRET, identity=identity)
    token.add_grant(voice_grant)

    return {"token": token.to_jwt(), "identity": identity}


@router.post("/voice-client")
async def voice_webhook_client(request: Request, CallSid: str = Form(...)):
    """
    Twilio's Voice webhook for the browser-softphone flow — the
    /voice-client analogue of twilio_voice.py's voice_webhook(). Same
    signature verification and the same <Start><Stream> fork to the
    EXISTING /ws/twilio (main.py, untouched — customer audio -> STT ->
    Layer 3/4/5 is fully reused, not duplicated), but bridges to
    <Client>{identity}</Client> (the agent's browser) instead of a real
    AGENT_PHONE_NUMBER phone call, so this leg never needs a verified number.
    """
    await _verify_twilio_signature(request)

    session_id = CallSid
    identity = request.query_params.get("identity") or _DEFAULT_IDENTITY
    logger.info(f"Twilio call {session_id}: browser-softphone leg -> Client({identity})")

    vr = VoiceResponse()

    start = Start()
    start.stream(url=f"{_ws_base_url()}/ws/twilio?session_id={session_id}", track="inbound_track")
    vr.append(start)

    dial = Dial()
    dial.client(identity)
    vr.append(dial)

    return Response(content=str(vr), media_type="application/xml")


@router.post("/call-client")
async def place_outbound_call_client(to: str = Form(...), identity: str = Form(default=_DEFAULT_IDENTITY)):
    """
    Browser-softphone analogue of twilio_voice.py's place_outbound_call —
    triggers an outbound call to a customer, but routes the answered call to
    /voice-client (this file) instead of /twilio/voice, so the agent side
    bridges to a Twilio.Device registered in the browser (identity must
    match what was passed to GET /token) rather than AGENT_PHONE_NUMBER.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        raise HTTPException(500, "TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_PHONE_NUMBER must be set")
    if not PUBLIC_BASE_URL:
        raise HTTPException(500, "PUBLIC_BASE_URL is not set — see .env")

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    voice_url = f"{PUBLIC_BASE_URL}/twilio/voice-client?identity={identity}"
    call = client.calls.create(to=to, from_=TWILIO_PHONE_NUMBER, url=voice_url)

    logger.info(f"Outbound call (browser-softphone) placed: CallSid={call.sid} to={to} identity={identity}")
    return {"call_sid": call.sid, "session_id": call.sid, "identity": identity}
