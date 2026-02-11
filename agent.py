import logging
import os
import asyncio
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import VoiceAgent, VoiceAgentSession
from livekit.plugins import deepgram, groq, silero
from agent_logic import agent_executor  # Import the shared "brain"


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class LegalVoiceAgent(VoiceAgent):
    def __init__(self):
        super().__init__(
            stt=groq.STT(
                model="whisper-large-v3",
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            llm=groq.LLM(
                model="llama3-70b-8192", # Used by VoiceAgent for internal tasks
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            tts=deepgram.TTS(
                model="aura-2-athena-en",
                api_key=os.getenv("DEEPGRAM_API_KEY"),
            ),
            vad=silero.VAD.load(),
        )
        self.agent_executor = agent_executor

    async def run(self, session: VoiceAgentSession):
        """This is the main loop for the voice agent."""
        logger.info("Voice agent session started.")
        await session.say("Namaste! I am your AI legal assistant. How can I help you today?")
        
        async for transcript in session.stt.stream():
            if not transcript.final:
                continue

            user_query = transcript.text
            logger.info(f"User said: {user_query}")
            
            if not user_query.strip():
                continue

            try:
                # Send query to the central "brain"
                # We use astream_events to stream the final response
                response_stream = self.agent_executor.astream_events(
                    {
                        "input": user_query,
                        "case_id": "",  # Add case_id (empty for voice)
                        "agent_scratchpad": []
                    },
                    version="v1"
                )
                
                final_answer = ""
                async for event in response_stream:
                    kind = event["event"]
                    
                    # Stream the final answer as it's generated
                    if kind == "on_chain_end":
                        data = event["data"]
                        if data.get("name") == "generate_final_answer":
                            final_answer = data["output"]["summary"]
                
                if final_answer:
                    logger.info(f"Agent response: {final_answer[:100]}...")
                    await session.say(final_answer)
                else:
                    await session.say("I'm sorry, I couldn't find an answer.")
                
                await session.say_flush()

            except Exception as e:
                logger.error(f"Error during agent execution: {e}")
                await session.say("I'm sorry, I encountered an error. Could you please repeat that?")
                await session.say_flush()

async def entrypoint(ctx: JobContext):
    """Worker entrypoint."""
    logger.info("Starting LegalVoiceAgent worker.")
    await ctx.connect()
    agent = LegalVoiceAgent()
    session = VoiceAgentSession(agent, ctx.room)
    await session.run()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))