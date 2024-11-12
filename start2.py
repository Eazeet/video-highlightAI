import os
import re
import json
import subprocess
from moviepy.editor import *
from moviepy.video.fx import crop, resize
import os
from dotenv import load_dotenv
from datetime import datetime
import httpx
from pydantic import BaseModel, Field
from typing import List
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
# import cv2
import pprint
pp = pprint.PrettyPrinter(indent=4)
from openai import OpenAI
from google.cloud import speech

load_dotenv()
deepgram_api_key = os.getenv("DEEPGRAM_API__KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)


deepgram_client = DeepgramClient(deepgram_api_key, DeepgramClientOptions(verbose=False))

def clean_json_output(raw_output):
    cleaned_output = re.sub(r'```json\s*|\s*```', '', raw_output).strip()
    cleaned_output = re.sub(r',\s*([\]}])', r'\1', cleaned_output)  
    
    # Identify and remove incomplete entries
    # This regex finds entries that don't have a closing curly brace or are incomplete
    if not cleaned_output.endswith(']'):
        last_bracket_index = cleaned_output.rfind('}')
        if last_bracket_index != -1:
            # Remove the incomplete part after the last valid entry
            cleaned_output = cleaned_output[:last_bracket_index + 1] + ']'

    # Check for any other potential issues like unmatched braces
    if cleaned_output.count('{') != cleaned_output.count('}'):
        cleaned_output = re.sub(r'\{[^}]*$', '', cleaned_output)  # Remove unmatched opening braces
    if cleaned_output.count('[') != cleaned_output.count(']'):
        cleaned_output += ']'  # Ensure there's a closing bracket for the list

    # Attempt to load the cleaned output as JSON
    try:
        clips = json.loads(cleaned_output)
        return clips
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Cleaned output:", cleaned_output)  # Print the cleaned output for debugging
        return None

# Download and extract audio from YouTube video
def download_audio(url):
    output_file = "output.m4a"
    if os.path.exists(output_file):
        os.remove(output_file)
    subprocess.call(["yt-dlp", "-x", "-f", "m4a", "-o", output_file, url])
    return output_file

# Download and extract video from YouTube
def download_video(url):
    output_file = "output.mp4"
    if os.path.exists(output_file):
        os.remove(output_file)
    subprocess.call(["yt-dlp", "-x", "-f", "mp4", "-o", output_file, url, '-k'])
    return output_file

# Convert audio to WAV format
def convert_to_wav(input_file, output_file):
    clip = AudioFileClip(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    clip.write_audiofile(output_file, codec='pcm_s16le', ffmpeg_params=['-ac', '1', '-ar', '16000'])

# Transcribe the audio using whisper
def transcribe_audio(audio_file):
    try:
        # Read audio file data
        with open(audio_file, "rb") as audio_file:
            buffer_data = audio_file.read()
        
        # Define audio source and transcription options
        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
            sentiment=True,
            diarize=True,
        )

        # Perform transcription
        before = datetime.now()
        response = deepgram_client.listen.rest.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )
        after = datetime.now()

        # Extract response details
        transcript_string = response.to_dict()["results"]["channels"][0]["alternatives"][0]["transcript"]
        words_list = response.to_dict()["results"]["channels"][0]["alternatives"][0]["words"]
        
        words_list = [
            {"word": word_info["word"], "start": word_info["start"], "end": word_info["end"]}
            for word_info in words_list
        ]

        sentence_list = [
        {
            "sentence": utterance["transcript"],
            "start": utterance["start"],
            "end": utterance["end"],
            "sentiment": utterance.get("sentiment", "neutral")  # Optional sentiment
        }
        for utterance in response.to_dict().get("results", {}).get("utterances", [])
    ]

        print(transcript_string)
        print("-----------------------------------")
        print(words_list)
        print("-----------------------------------")
        print(sentence_list)
        # Return words list and full transcript
        return words_list, transcript_string, sentence_list
    
    except Exception as e:
        print(f"Exception occurred during transcription: {e}")
        return [], ""

# Process the transcript and identify key points
def process_transcript(transcript):
    # Placeholder, replace with your implementation
    processed_content = transcript
    return processed_content

# Split the content into tweets
def split_into_tweets(content):
    tweets = []
    tweet_limit = 280

    while len(content) > tweet_limit:
        tweet = content[:tweet_limit]
        tweets.append(tweet)
        content = content[tweet_limit:]

    if len(content) > 0:
        tweets.append(content)

    return tweets

# Cut and join the video
def cut_video(video, start_time, end_time):
    cut_video = video.subclip(start_time, end_time)
    return cut_video

# Cut and join the video
def cut_and_join_video(video, times):
    clips = []
    for time in times:
        start = time["start"]
        stop = time["end"]
        clip = cut_video(video, start, stop)
        clips.append(clip)
    joined_video = concatenate_videoclips(clips)
    return joined_video 

def get_keywords(text):
    prompt = """Using the text provided, output the keywords seperated by commas
    Example input: Africa is a fascinating continent known for its incredible diversity and abundant wildlife.
    It is a land of breathtaking landscapes, vibrant cultures, and rich history.
    The African continent is the second-largest in the world, covering a vast area
    with diverse ecosystems. From the vast Sahara Desert in the north to the lush
    rainforests of Central Africa and the stunning savannahs of the Serengeti, Africa
    offers a wide range of natural wonders. 🐘 Africa is home to some of the most iconic
    wildlife species on the planet. The majestic lions, graceful giraffes, powerful
    elephants, and elusive leopards are just a few examples of the incredible
    biodiversity found here.

    Example output: Africa,continent,diverse,wildlife,cultures,elephant

    Input: """ + str(text) + "\n\nOutput:"
    # Create the prompt for extracting keywords
    messages = [
        {"role": "system", "content": "You're a very helpful assistant"},
        {"role": "user", "content": prompt}
    ]

    # Generate keywords using the chat completion method
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )

    # Process and return the keywords
    result = completion.choices[0].message.content
    return result.split(',')

def get_highlight(content):
    prompt = (
        "Using the data provided, select clips to create a highlight video lasting 30-45 seconds"
        "The input is a list of sentences, each with fields: 'sentence' for the spoken text, "
        "'start' and 'end' for timestamps, and 'sentiment' with values 'positive', 'negative', or 'neutral'. "
        "Include all of the clips with high sentiment (either positive or negative). "
        "Do not completely exclude the ones marked as neutral sentiment, include some that you think might make sense to add, just to get the video up to about 30 seconds"
        "Make the video 30 seconds long by choosing all possible sentences."
        "The output format should be a JSON array of objects, each with 'start' and 'end' times for the selected clips.\n\n"
        "Do not output anything other than the JSON data. JSON data only!\n"
        "Input: " + str(content) + "\n\nOutput:"
    )
    
    messages = [
        {"role": "system", "content": "You're a very helpful assistant"},
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    raw_output = completion.choices[0].message.content
    print("Raw output:", raw_output)
    clips = clean_json_output(raw_output)
    if clips is not None:
        print("Parsed clips:", clips)
    else:
        print("Failed to parse clips.")

    return clips

def get_clip_times(content, clips):
    clip_times = []
    for i in clips:
        if i >= 0 and i < len(content):
            start = content[i]["start"]
            end = content[i]["end"]
            clip_times.append([start, end])
    return clip_times

def get_captions(content, clips):
    content = [] + content # this makes a copy
    captions = []
    # for i in clips:
    #     if i >= 0 and i < len(content):
    #         captions.append(content[i])
    for time in clips:
        words = []
        start = time['start']
        end = time['end']
        in_clip = False
        for c in content:
            if not in_clip and c['start'] >= start:
                in_clip = True
            if in_clip:
                words.append(c["sentence"])
            if in_clip and c['end'] >= end:
                in_clip = False
                break
        text = ' '.join(words)
        captions.append({"start": start, 'end':end, "text":text})
    
    print("captions: ", captions)
    return captions

def create_word_clips(words, font_name, font_size, video_width, video_height):
    """Create individual TextClips for each word with precise timing."""
    word_clips = []
    current_line = []
    current_line_width = 0
    line_height = font_size * 1.5
    max_width = video_width - 40  # Leaving margins
    y_position = video_height - 150  # Base position for text
    
    # First pass: organize words into lines
    lines = []
    current_line = []
    current_line_width = 0
    
    for word_info in words:
        # Create a temporary clip to measure text width
        word_clip = TextClip(
            word_info["word"],
            font=font_name,
            font_size=font_size,
            color='white',
            stroke_color='black',
            stroke_width=2
        )
        
        word_width = word_clip.w
        word_clip.close()
        
        if current_line_width + word_width > max_width:
            lines.append(current_line)
            current_line = []
            current_line_width = 0
        
        current_line.append((word_info, word_width))
        current_line_width += word_width + font_size/2  # Add space between words
    
    if current_line:
        lines.append(current_line)
    
    # Second pass: create clips with proper positioning
    for line_idx, line in enumerate(lines):
        line_width = sum(word[1] for word in line) + (len(line) - 1) * (font_size/2)
        x_position = (video_width - line_width) / 2
        
        for word_info, word_width in line:
            # Create background for better readability
            bg_clip = ColorClip(
                size=(word_width + 10, int(line_height)),
                color=(0, 0, 0, 0.5)
            ).with_position(
                (x_position - 5, y_position + (line_idx * line_height))
            ).with_start(
                word_info["start"]
            ).with_duration(
                word_info["end"] - word_info["start"]
            ).with_opacity(0.7)
            
            # Create word clip
            word_clip = TextClip(
                word_info["word"],
                font=font_name,
                font_size=font_size,
                color='white',
                stroke_color='black',
                stroke_width=2
            ).with_position(
                (x_position, y_position + (line_idx * line_height))
            ).with_start(
                word_info["start"]
            ).with_duration(
                word_info["end"] - word_info["start"]
            )
            
            word_clips.extend([bg_clip, word_clip])
            x_position += word_width + font_size/2
    
    return word_clips

def get_words_in_clip(words_list, clip_start, clip_end):
    """Filter words that fall within the clip timeframe."""
    return [
        word for word in words_list 
        if word["start"] >= clip_start 
        and word["end"] <= clip_end
    ]

def draw_on_video(video, captions, words_list, font_name, font_size, logo, keywords):
    """Create video with animated word-by-word captions."""
    all_clips = [video]
    for caption in captions:
        segment_words = get_words_in_clip(words_list, caption["start"], caption["end"])
        
        # Create word clips for this segment
        word_clips = create_word_clips(
            segment_words,
            font_name,
            font_size,
            video.w,
            video.h
        )
        
        all_clips.extend(word_clips)
    
    # If there's a logo, add it
    if logo:
        # Add logo implementation here
        pass
    final_video = CompositeVideoClip(all_clips)
    return final_video

# Main function
def main():
    #  uncomment this if you want to see the list of available fonts
    # pp.pprint(TextClip.list("font"))



    youtube_url = input("Enter the YouTube video URL: ")
    audio_file = download_audio(youtube_url)
    video_file = download_video(youtube_url)
    wav_file = "output.wav"
    convert_to_wav(audio_file, wav_file)
    words_list, transcript_string, sentence_list = transcribe_audio(wav_file)
    num_vids = int(input("Enter the number of highlight videos to generate: "))
    default_font = "American-Captain"
    font_name = input("Enter the font name (Default: "+default_font+"): ").strip()
    if font_name == "":
        font_name = default_font
    logo_path = input("Enter the path to the logo image file: ")
    font_scale = int(input("Enter the font size: "))

    for i in range(num_vids):
        # Process the transcript and identify key points
        processed_content = process_transcript(sentence_list)
        
        # get keywords
        keywords = get_keywords(transcript_string)
        print("Keywords: ", keywords)

        # Generate captions with emojis
        clips = get_highlight(processed_content)
        print('CLIPS', clips ) 
        # clip_times = get_clip_times(processed_content, clips)
        # print(clip_times)

        # Get the video file
        video_file = "output.mp4"
        video = VideoFileClip(video_file)
        print("Old size: ", video.size)
        video = resize(video, height=1920)
        video = crop(video, y_center=960, x_center=video.w/2, height=1920, width=1080)
        print("New size: ", video.size)

        # Draw captions onto the video
        captions = get_captions(processed_content, clips)
        video_captioned = draw_on_video(video, captions, words_list, font_name, font_scale, logo_path, keywords)

        # Cut and join the video
        highlights_video = cut_and_join_video(video_captioned, clips)

        output_file = f"highlight{i}.mp4"
        highlights_video.write_videofile(output_file)

        print("Captioned video created:", output_file)

    # except Exception as e:
    #     print("An error occurred:")
    #     print(str(e))

# Execute the main function
if __name__ == "__main__":
    main()