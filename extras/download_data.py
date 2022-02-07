'''
Download Office YouTube videos
'''
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process
from pytube import YouTube
import youtube_dl

nthreads = 4
saved_videos_dir = 'saved_videos'
os.makedirs(saved_videos_dir, exist_ok=True)
yt_download_template = "yt-dlp -f \"best\" https://www.youtube.com/watch?v={} -o '{}/%(id)s.%(ext)s'"

# Download youtube video given the youtube id
def download_video(video_thread):
    video_id, thread_id = video_thread
    print(f'Processing {video_id} using {thread_id}')
    os.system(yt_download_template.format(video_id, saved_videos_dir))

if __name__ == '__main__':
    video_ids = ['-VwgK3V938E', 'zTYSTk8iLsM', 'nxD1LAOabAw']
    
    # multiprocess code to download multiple videos at once using threads
    p = ThreadPoolExecutor(nthreads)

    print(f'Videos to download : {video_ids}')

    jobs = [(video_file, job_id%nthreads) for job_id, video_file in enumerate(video_ids)]
    futures = [p.submit(download_video, job) for job in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]