import gradio as gr
import tempfile
import os
from typing import Tuple
import yt_dlp

class VideoProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def process_video(
        self,
        youtube_url: str,
        num_highlights: int,
        font_name: str = "American-Captain",
        font_size: int = 48,
        logo_file: str = None,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """Process video and create highlights with captions."""
        try:
            progress(0, desc="Downloading video...")
            # Download video and audio
            audio_file = self.download_audio(youtube_url)
            video_file = self.download_video(youtube_url)
            
            progress(0.2, desc="Converting audio...")
            # Convert to WAV
            wav_file = os.path.join(self.temp_dir, "output.wav")
            convert_to_wav(audio_file, wav_file)
            
            progress(0.3, desc="Transcribing audio...")
            # Transcribe
            words_list, transcript_string, sentence_list = transcribe_audio(wav_file)
            
            output_files = []
            for i in range(num_highlights):
                progress(0.4 + (i * 0.2), desc=f"Creating highlight {i+1}...")
                
                # Process transcript and get keywords
                processed_content = process_transcript(sentence_list)
                keywords = get_keywords(transcript_string)
                
                # Generate clips and captions
                clips = get_highlight(processed_content)
                
                # Process video
                video = VideoFileClip(video_file)
                video = resize(video, height=1920)
                video = crop(video, y_center=960, x_center=video.w/2, height=1920, width=1080)
                
                # Add captions
                captions = get_captions(processed_content, clips)
                video_captioned = draw_on_video(
                    video, 
                    captions, 
                    words_list,
                    font_name, 
                    font_size, 
                    logo_file, 
                    keywords
                )
                
                # Create final video
                highlights_video = cut_and_join_video(video_captioned, clips)
                
                # Save output
                output_file = os.path.join(self.temp_dir, f"highlight_{i}.mp4")
                highlights_video.write_videofile(output_file)
                output_files.append(output_file)
                
                # Clean up
                video.close()
                video_captioned.close()
                highlights_video.close()
            
            progress(1.0, desc="Complete!")
            return output_files, transcript_string
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def download_audio(self, url: str) -> str:
        """Download audio from YouTube URL."""
        output_file = os.path.join(self.temp_dir, "output.m4a")
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': output_file,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_file
    
    def download_video(self, url: str) -> str:
        """Download video from YouTube URL."""
        output_file = os.path.join(self.temp_dir, "output.mp4")
        ydl_opts = {
            'format': 'mp4/bestvideo/best',
            'outtmpl': output_file,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_file

def create_interface():
    """Create and launch the Gradio interface."""
    processor = VideoProcessor()
    
    def process_inputs(
        youtube_url: str,
        num_highlights: int,
        font_name: str,
        font_size: int,
        logo_file: str = None,
        progress=gr.Progress()
    ) -> Tuple[list, str]:
        return processor.process_video(
            youtube_url,
            num_highlights,
            font_name,
            font_size,
            logo_file,
            progress
        )
    
    # Create the interface
    with gr.Blocks(title="Video Highlight Generator") as interface:
        gr.Markdown("""
        # Video Highlight Generator
        Upload a YouTube video URL and generate highlight clips with synchronized captions.
        """)
        
        with gr.Row():
            with gr.Column():
                # Input components
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="Enter YouTube video URL here..."
                )
                num_highlights = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Highlights to Generate"
                )
                font_name = gr.Dropdown(
                    choices=["American-Captain", "Arial", "Helvetica"],
                    value="American-Captain",
                    label="Font Name"
                )
                font_size = gr.Slider(
                    minimum=24,
                    maximum=72,
                    value=48,
                    step=2,
                    label="Font Size"
                )
                logo_file = gr.File(
                    label="Logo (optional)",
                    type="filepath"
                )
                
                # Process button
                process_btn = gr.Button("Generate Highlights", variant="primary")
            
            with gr.Column():
                # Output components
                output_videos = gr.File(
                    label="Generated Highlights",
                    file_count="multiple"
                )
                transcript = gr.Textbox(
                    label="Transcript",
                    lines=10,
                    readonly=True
                )
        
        # Set up event handler
        process_btn.click(
            fn=process_inputs,
            inputs=[
                youtube_url,
                num_highlights,
                font_name,
                font_size,
                logo_file
            ],
            outputs=[
                output_videos,
                transcript
            ]
        )
        
        # Example usage
        gr.Examples(
            examples=[
                ["https://www.youtube.com/watch?v=example1", 1, "American-Captain", 48, None],
                ["https://www.youtube.com/watch?v=example2", 2, "Arial", 36, None]
            ],
            inputs=[youtube_url, num_highlights, font_name, font_size, logo_file]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )