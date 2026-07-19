import asyncio
from dotenv import load_dotenv
load_dotenv()

from layer4 import TriggerGate, ToolRouter, get_router_client, build_default_registry

async def main():
    prompt = "what are the different types of pension policies available in lic"

    gate = TriggerGate(session_id="test")
    router = ToolRouter(
        session_id="test",
        llm_client=get_router_client(),
        registry=build_default_registry(),
        fallback_gate=gate,
    )

    decision = await router.route(speaker="agent", text=prompt, context="", now=0.0)
    print(decision)

asyncio.run(main())
