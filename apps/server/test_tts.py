import asyncio
import httpx
import json
import base64

TTS_BASE_URL = "http://192.168.1.111:8884/dev"


async def test_tts():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{TTS_BASE_URL}/captioned_speech",
            json={
                "model": "kokoro",
                "input": "Hello world, this is a test.",
                "voice": "af_heart",
                "speed": 1.0,
                "response_format": "pcm",
                "stream": True,
            },
        )

        print(f"Status: {response.status_code}")

        chunk_count = 0
        async for line in response.aiter_lines():
            if line.strip():
                chunk_count += 1
                try:
                    data = json.loads(line)
                    print(f"\n{'='*60}")
                    print(f"Chunk #{chunk_count}")
                    print(f"{'='*60}")
                    print(f"Keys: {list(data.keys())}")

                    if "audio" in data:
                        audio_len = len(base64.b64decode(data["audio"]))
                        print(f"Audio: {audio_len} bytes")

                    if "timestamps" in data:
                        timestamps = data["timestamps"]
                        print(f"Timestamps: {len(timestamps)} words")
                        if timestamps:
                            print(f"First: {timestamps[0]}")
                            print(f"Last: {timestamps[-1]}")
                    else:
                        print("❌ NO TIMESTAMPS FIELD!")

                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"Raw line: {line[:100]}")


if __name__ == "__main__":
    asyncio.run(test_tts())
