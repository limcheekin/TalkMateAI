import asyncio
import json
import base64
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List
import uvicorn
import re

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# OpenAI client
from openai import AsyncOpenAI
import httpx
import io, wave

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://192.168.1.111:8880/v1"
TTS_BASE_URL = "http://192.168.1.111:8884/dev"
VLM_MODEL = "smolvlm2-500m-video-instruct"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "kokoro"
TTS_VOICE = "af_heart"


class WordTimingEstimator:
    """Estimates word-level timing for synthesized speech (fallback method)"""
    
    def __init__(self):
        self.avg_word_duration_ms = 400
        self.min_word_duration_ms = 100
        self.max_word_duration_ms = 1000
        
        self.pause_after_punctuation = {
            '.': 300,
            '!': 300,
            '?': 300,
            ',': 150,
            ';': 200,
            ':': 200,
        }
    
    def estimate_word_timings(self, text: str, audio_duration_ms: float) -> List[Dict[str, Any]]:
        """Estimate word-level timing based on text and audio duration"""
        tokens = self._tokenize_with_punctuation(text)
        
        if not tokens:
            return []
        
        token_weights = []
        total_weight = 0
        
        for token in tokens:
            if token.strip() in self.pause_after_punctuation:
                weight = 0.1
            else:
                weight = max(0.5, len(token) / 5.0)
            
            token_weights.append(weight)
            total_weight += weight
        
        timings = []
        current_time = 0.0
        
        for i, (token, weight) in enumerate(zip(tokens, token_weights)):
            if token.strip() in self.pause_after_punctuation:
                pause_duration = self.pause_after_punctuation[token.strip()]
                current_time += pause_duration
                continue
            
            word_duration = (weight / total_weight) * audio_duration_ms
            word_duration = max(self.min_word_duration_ms, 
                              min(self.max_word_duration_ms, word_duration))
            
            start_time = current_time
            end_time = current_time + word_duration
            
            timings.append({
                "word": token,
                "start_time": round(start_time, 1),
                "end_time": round(end_time, 1)
            })
            
            current_time = end_time
        
        if timings and current_time > 0:
            scale_factor = audio_duration_ms / current_time
            for timing in timings:
                timing["start_time"] = round(timing["start_time"] * scale_factor, 1)
                timing["end_time"] = round(timing["end_time"] * scale_factor, 1)
        
        return timings
    
    def _tokenize_with_punctuation(self, text: str) -> List[str]:
        """Tokenize text into words and punctuation marks"""
        pattern = r"(\w+|[.,!?;:])"
        tokens = re.findall(pattern, text)
        return [t for t in tokens if t.strip()]


class ImageManager:
    """Manages image saving and verification"""

    def __init__(self, save_directory="received_images"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        logger.info(f"Image save directory: {self.save_directory.absolute()}")

    def save_image(self, image_data: bytes, client_id: str, prefix: str = "img") -> Optional[str]:
        """Save image data and return the filename"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}_{client_id}_{timestamp}.jpg"
            filepath = self.save_directory / filename

            with open(filepath, "wb") as f:
                f.write(image_data)

            file_size = len(image_data)
            logger.info(f"💾 Saved image: {filename} ({file_size:,} bytes)")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error saving image: {e}")
            return None


class OpenAIProcessor:
    """Unified processor using OpenAI-compatible APIs"""

    def __init__(self, base_url: str, dev_base_url: str):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy-key",
            timeout=httpx.Timeout(60.0, connect=10.0)
        )
        self.dev_base_url = dev_base_url
        self.message_history: list = []
        self.max_history_messages = 8
        self.current_image_b64: Optional[str] = None
        self.lock = asyncio.Lock()
        
        # Counters
        self.transcription_count = 0
        self.generation_count = 0
        self.tts_count = 0
        
        # Fallback timing estimator
        self.timing_estimator = WordTimingEstimator()
        
        logger.info("OpenAI processor initialized")

    def raw_pcm16_to_wav_file(self, raw_bytes, sample_rate=16000, nchannels=1):
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(2)          # 16-bit => 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(raw_bytes)
        bio.seek(0)
        bio.name = "audio.wav"
        return bio

    async def transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio using Whisper API"""
        try:
            response = await self.client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=self.raw_pcm16_to_wav_file(audio_bytes),
                response_format="verbose_json"
            )
            
            transcribed_text = response.strip() if isinstance(response, str) else response.text.strip()
            self.transcription_count += 1
            
            logger.info(f"Transcription #{self.transcription_count}: '{transcribed_text}'")
            
            if not transcribed_text or len(transcribed_text) < 3:
                return "NO_SPEECH"
            
            noise_indicators = ["thank you", "thanks for watching", "you", "."]
            if transcribed_text.lower().strip() in noise_indicators:
                return "NOISE_DETECTED"
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def set_image(self, image_data: bytes) -> bool:
        """Cache the most recent image as base64"""
        async with self.lock:
            try:
                self.current_image_b64 = base64.b64encode(image_data).decode('utf-8')
                self.message_history = []
                logger.info("Image cached successfully as base64")
                return True
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return False

    async def generate_response_stream(
        self, 
        user_text: str,
        include_image: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using SmolVLM2 API"""
        async with self.lock:
            try:
                messages = self.message_history.copy()
                
                if include_image and self.current_image_b64:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.current_image_b64}"
                                }
                            },
                            {"type": "text", "text": user_text}
                        ]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": user_text
                    })
                
                stream = await self.client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=messages,
                    max_tokens=1200,
                    stream=True,
                    temperature=0.7
                )
                
                self.generation_count += 1
                full_response = ""
                
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            full_response += delta.content
                            yield delta.content
                
                self.message_history.append({
                    "role": "user",
                    "content": user_text
                })
                self.message_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                if len(self.message_history) > self.max_history_messages:
                    self.message_history = self.message_history[-self.max_history_messages:]
                
                logger.info(f"Generation #{self.generation_count} complete: {len(full_response)} chars")
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                yield f"Error: {str(e)}"

    async def synthesize_speech_with_timing(
        self, 
        text: str,
        use_native_timing: bool = True
    ) -> tuple[Optional[bytes], List[Dict[str, Any]]]:
        """Synthesize speech with word-level timing
        
        Returns:
            Tuple of (audio_bytes, word_timings)
        """
        try:
            logger.info(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            if use_native_timing:
                # Use custom captioned_speech endpoint for native timing
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{self.dev_base_url}/captioned_speech",
                            json={
                                "model": TTS_MODEL,
                                "input": text,
                                "voice": TTS_VOICE,
                                "speed": 1.0,
                                "response_format": "pcm",
                                "stream": True,
                            },
                        )
                        response.raise_for_status()
                        
                        all_audio_chunks = []
                        all_word_timings = []
                        
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk_data = json.loads(line)
                                    
                                    # Decode audio
                                    if "audio" in chunk_data:
                                        audio_bytes = base64.b64decode(chunk_data["audio"])
                                        all_audio_chunks.append(audio_bytes)
                                    
                                    # Extract timestamps
                                    if "timestamps" in chunk_data and chunk_data["timestamps"]:
                                        for ts in chunk_data["timestamps"]:
                                            word_timing = {
                                                "word": ts.get("word", ts.get("text", "")),
                                                "start_time": ts.get("start", 0) * 1000,
                                                "end_time": ts.get("end", 0) * 1000,
                                            }
                                            all_word_timings.append(word_timing)
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse chunk: {e}")
                                    continue
                        
                        if all_audio_chunks:
                            combined_audio = b''.join(all_audio_chunks)
                            self.tts_count += 1
                            
                            logger.info(
                                f"✨ TTS #{self.tts_count} complete: {len(combined_audio):,} bytes, "
                                f"{len(all_word_timings)} word timings (NATIVE)"
                            )
                            
                            return combined_audio, all_word_timings
                        else:
                            logger.warning("No audio chunks received from native endpoint")
                            raise Exception("No audio data received")
                            
                except Exception as e:
                    logger.warning(f"Native timing failed: {e}, falling back to estimation")
                    use_native_timing = False
            
            # Fallback to standard endpoint with estimated timing
            if not use_native_timing:
                response = await self.client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=text,
                    response_format="pcm",
                    speed=1.0
                )
                
                audio_chunks = []
                async for chunk in response.iter_bytes(chunk_size=8192):
                    audio_chunks.append(chunk)
                
                if not audio_chunks:
                    logger.error("No audio chunks received from standard endpoint")
                    return None, []
                
                combined_audio = b''.join(audio_chunks)
                total_bytes = len(combined_audio)
                
                # Calculate duration (24kHz, 16-bit PCM)
                sample_rate = 24000
                bytes_per_sample = 2
                total_samples = total_bytes / bytes_per_sample
                audio_duration_ms = (total_samples / sample_rate) * 1000
                
                word_timings = self.timing_estimator.estimate_word_timings(text, audio_duration_ms)
                
                self.tts_count += 1
                logger.info(
                    f"✨ TTS #{self.tts_count} complete: {total_bytes:,} bytes, "
                    f"{len(word_timings)} word timings (estimated)"
                )
                
                return combined_audio, word_timings
            
            return None, []
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, []


class ConnectionManager:
    """Manages WebSocket connections and processing tasks"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.current_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        self.image_manager = ImageManager()
        self.stats = {
            "audio_segments_received": 0,
            "images_received": 0,
            "audio_with_image_received": 0,
            "last_reset": datetime.now(),
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.current_tasks[client_id] = {"processing": None}
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.current_tasks:
            del self.current_tasks[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def cancel_current_tasks(self, client_id: str):
        """Cancel any ongoing processing tasks for a client"""
        if client_id in self.current_tasks:
            tasks = self.current_tasks[client_id]
            
            for task_type, task in tasks.items():
                if task and not task.done():
                    logger.info(f"Cancelling {task_type} task for client {client_id}")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.current_tasks[client_id] = {"processing": None}

    def set_task(self, client_id: str, task_type: str, task: asyncio.Task):
        """Set a task for a client"""
        if client_id in self.current_tasks:
            self.current_tasks[client_id][task_type] = task

    def update_stats(self, event_type: str):
        """Update statistics"""
        if event_type in self.stats:
            self.stats[event_type] += 1

    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = datetime.now() - self.stats["last_reset"]
        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "active_connections": len(self.active_connections),
        }


manager = ConnectionManager()
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global processor
    
    logger.info("Initializing OpenAI processor...")
    try:
        processor = OpenAIProcessor(API_BASE_URL, TTS_BASE_URL)
        logger.info("OpenAI processor initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing processor: {e}")
        raise

    yield

    logger.info("Shutting down server...")
    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
        manager.disconnect(client_id)
    logger.info("Server shutdown complete")


app = FastAPI(
    title="Voice Assistant with OpenAI APIs",
    description="Real-time voice assistant using OpenAI-compatible APIs",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    stats = manager.get_stats()
    if processor:
        stats.update({
            "transcriptions": processor.transcription_count,
            "generations": processor.generation_count,
            "tts_calls": processor.tts_count,
        })
    return stats


@app.get("/images")
async def list_saved_images():
    """List all saved images"""
    try:
        images_dir = manager.image_manager.save_directory
        if not images_dir.exists():
            return {"images": [], "message": "No images directory found"}

        images = []
        for image_file in images_dir.glob("*.jpg"):
            stat = image_file.stat()
            images.append({
                "filename": image_file.name,
                "path": str(image_file),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            })

        images.sort(key=lambda x: x["created"], reverse=True)
        return {"images": images, "count": len(images)}

    except Exception as e:
        logger.error(f"Error listing images: {e}")
        return {"error": str(e)}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time multimodal interaction"""
    await manager.connect(websocket, client_id)

    try:
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

        async def process_audio_segment(audio_data: bytes, image_data: Optional[bytes] = None):
            """Process audio segment through the complete pipeline"""
            try:
                if image_data:
                    logger.info(f"🎥 Processing audio+image: audio={len(audio_data)} bytes, image={len(image_data)} bytes")
                    manager.update_stats("audio_with_image_received")
                    
                    saved_path = manager.image_manager.save_image(image_data, client_id, "multimodal")
                    if saved_path:
                        logger.info(f"📸 Image saved: {saved_path}")
                    
                    await processor.set_image(image_data)
                else:
                    logger.info(f"🎤 Processing audio-only: {len(audio_data)} bytes")
                    manager.update_stats("audio_segments_received")

                # Send interrupt signal
                await websocket.send_text(json.dumps({"interrupt": True}))

                # Step 1: Transcribe audio
                logger.info("Starting transcription...")
                transcribed_text = await processor.transcribe_audio(audio_data)
                
                if transcribed_text in ["NOISE_DETECTED", "NO_SPEECH", None]:
                    logger.info(f"Skipping processing: {transcribed_text}")
                    return

                # Step 2: Generate response stream
                logger.info("Starting response generation...")
                response_stream = processor.generate_response_stream(
                    transcribed_text,
                    include_image=bool(image_data)
                )

                # Collect initial text for first TTS
                initial_text = ""
                min_chars = 50
                sentence_endings = {'.', '!', '?'}
                
                async for chunk in response_stream:
                    initial_text += chunk
                    
                    if len(initial_text) >= min_chars:
                        if any(initial_text.endswith(end) for end in sentence_endings):
                            break
                        if len(initial_text) >= min_chars * 1.5 and ',' in initial_text:
                            break
                    
                    if len(initial_text) >= min_chars * 3:
                        break

                # Step 3: Generate TTS for initial text
                if initial_text:
                    logger.info(f"Generating TTS for initial text: '{initial_text[:30]}...'")
                    
                    audio_data, word_timings = await processor.synthesize_speech_with_timing(initial_text)
                    
                    if audio_data:
                        base64_audio = base64.b64encode(audio_data).decode('utf-8')
                        
                        audio_message = {
                            "audio": base64_audio,
                            "word_timings": word_timings,
                            "sample_rate": 24000,
                            "method": "native_kokoro_timing",
                            "modality": "multimodal" if image_data else "audio_only",
                        }
                        
                        await websocket.send_text(json.dumps(audio_message))
                        logger.info(f"✨ Initial audio sent with {len(word_timings)} word timings")

                # Step 4: Process remaining text chunks
                remaining_text = ""
                async for chunk in response_stream:
                    remaining_text += chunk
                    
                    if len(remaining_text) >= 80 and any(remaining_text.endswith(end) for end in sentence_endings):
                        audio_data, word_timings = await processor.synthesize_speech_with_timing(remaining_text)
                        
                        if audio_data:
                            base64_audio = base64.b64encode(audio_data).decode('utf-8')
                            
                            chunk_audio_message = {
                                "audio": base64_audio,
                                "word_timings": word_timings,
                                "sample_rate": 24000,
                                "method": "native_kokoro_timing",
                                "chunk": True,
                                "modality": "multimodal" if image_data else "audio_only",
                            }
                            
                            await websocket.send_text(json.dumps(chunk_audio_message))
                            logger.info(f"✨ Chunk audio sent with {len(word_timings)} word timings")
                        
                        remaining_text = ""

                # Send any remaining text
                if remaining_text:
                    audio_data, word_timings = await processor.synthesize_speech_with_timing(remaining_text)
                    
                    if audio_data:
                        base64_audio = base64.b64encode(audio_data).decode('utf-8')
                        
                        final_audio_message = {
                            "audio": base64_audio,
                            "word_timings": word_timings,
                            "sample_rate": 24000,
                            "method": "native_kokoro_timing",
                            "chunk": True,
                            "modality": "multimodal" if image_data else "audio_only",
                        }
                        
                        await websocket.send_text(json.dumps(final_audio_message))
                        logger.info(f"✨ Final chunk audio sent with {len(word_timings)} word timings")

                # Signal completion
                await websocket.send_text(json.dumps({"audio_complete": True}))
                logger.info("Audio processing complete")

            except asyncio.CancelledError:
                logger.info("Audio processing cancelled")
                raise
            except Exception as e:
                logger.error(f"Error processing audio segment: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")

        async def receive_and_process():
            """Receive and process messages from the client"""
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if "audio_segment" in message:
                        await manager.cancel_current_tasks(client_id)
                        
                        audio_data = base64.b64decode(message["audio_segment"])
                        image_data = None
                        
                        if "image" in message:
                            image_data = base64.b64decode(message["image"])
                            logger.info("Received audio+image segment")
                        else:
                            logger.info("Received audio-only segment")
                        
                        processing_task = asyncio.create_task(
                            process_audio_segment(audio_data, image_data)
                        )
                        manager.set_task(client_id, "processing", processing_task)

                    elif "image" in message:
                        if not (client_id in manager.current_tasks 
                                and manager.current_tasks[client_id]["processing"] 
                                and not manager.current_tasks[client_id]["processing"].done()):
                            image_data = base64.b64decode(message["image"])
                            manager.update_stats("images_received")
                            
                            saved_path = manager.image_manager.save_image(
                                image_data, client_id, "standalone"
                            )
                            if saved_path:
                                logger.info(f"📸 Standalone image saved: {saved_path}")
                            
                            await processor.set_image(image_data)
                            logger.info("Image updated")

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected during receive loop")
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

        async def send_keepalive():
            """Send periodic keepalive pings"""
            try:
                while True:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": datetime.now().timestamp()})
                    )
                    await asyncio.sleep(10)
            except Exception:
                pass

        receive_task = asyncio.create_task(receive_and_process())
        keepalive_task = asyncio.create_task(send_keepalive())

        done, pending = await asyncio.wait(
            [receive_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        logger.info(f"Cleaning up resources for client {client_id}")
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)


def main():
    """Main function to start the FastAPI server"""
    logger.info("Starting Voice Assistant server with OpenAI APIs...")

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=60,
        timeout_keep_alive=30,
    )

    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
