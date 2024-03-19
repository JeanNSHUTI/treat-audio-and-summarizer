import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import librosa
import librosa.display
import noisereduce as nr
import pydub
from scipy.io import wavfile
import csv
import pathlib
import mysql.connector
import smtplib, ssl

# Summarizer imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from PyPDF2 import PdfReader
from newspaper import Article


# output_path = r'C:\Users\Admin\Documents\HE2B\Master2\LangageScript\output'
local_db = 'localdb.csv'
header = ['OPath', 'File name', 'Loudness (dBFS)', 'Channels', 'Sample width (1=8B)', 'Frame Rate', 'RMS',
          'Max (dBFS)', 'Durations (s)', 'DestPath', 'Treated']


"""
    This section of code handles all audio treatment functions
"""


# Audio Job details
class AudioJob:
    def __init__(self):
        self.mode = "denoise"
        self.songs = []
        self.artists = []
        self.no_of_tracks = 0
        self.destination_path = r''

    def set_mode(self, cmode):
        if cmode == "True":
            self.mode = "convert"
        else:
            self.mode = "denoise"


def print_hi(name):
    print(f'Arg 0: {name}')


def create_local_csv():

    with open(local_db, 'w', newline="") as file:
        csvwriter = csv.writer(file)  # create a csvwriter object
        csvwriter.writerow(header)  # write the header
        # csvwriter.writerows(data)  # 5. write the rest of the data

    # Check if database has been created, if not create db
    initial_mysql_connect()
    create_initial_table()


def create_global_output_path(basepath):
    # Check if output directory exists, if not, create it
    new_basepath = pathlib.Path(basepath).parent
    output_path = os.path.join(new_basepath, "output")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    return output_path


def initial_mysql_connect():
    conn = mysql.connector.connect(
        host="localhost",
        user="root" ,
        password="56955jean"
    )
    db_cursor = conn.cursor()

    db_cursor.execute("SHOW DATABASES")
    if "langage_script" not in db_cursor.fetchall():
        sql_query = "CREATE DATABASE IF NOT EXISTS langage_script"
        db_cursor.execute(sql_query)


def connect_to_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root" ,
        password="56955jean",
        database="langage_script"
    )
    return conn


def create_initial_table():
    # Check if table exists, if not, create table
    dbconn = connect_to_db()
    db_cursor = dbconn.cursor()

    db_cursor.execute("SHOW TABLES")

    if "job" not in db_cursor.fetchall():
        sql_query = "CREATE TABLE IF NOT EXISTS JOB" \
                    "(JOBID INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY ," \
                    "DONE_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP  ," \
                    "SOURCE_PATH VARCHAR(255) NOT NULL  ," \
                    "MODE CHAR(10) NOT NULL  ," \
                    "NO_OF_FILES_TREATED INTEGER NOT NULL  ," \
                    "DEST_PATH VARCHAR(255) NOT NULL  ," \
                    "ARTISTS VARCHAR(255) NULL  ," \
                    "SONGS VARCHAR(255) NULL)"
        db_cursor.execute(sql_query.lower())
    dbconn.close()


def register_new_job(track_counter, o_path, dest_path, mode, artists, songs):
    dbconn = connect_to_db()
    db_cursor = dbconn.cursor()

    print(mode)
    print(' - '.join(artists))
    print(' - '.join(songs))
    print(track_counter)

    sql_query = "INSERT INTO job (source_path, mode, no_of_files_treated, dest_path, artists, songs)" \
                " VALUES (%s, %s, %s, %s, %s, %s)"
    row_values = (o_path, mode, track_counter, dest_path, ' - '.join(artists), ' - '.join(songs))
    db_cursor.execute(sql_query, row_values)

    dbconn.commit()

    if db_cursor.rowcount > 0:
        print("Successfully persisted row of data")
    else:
        print("Failed to insert to db")

    return db_cursor.lastrowid


def get_audio_filepaths(path):
    audio_file_paths = pathlib.Path(path)
    # Only return untreated audio file paths
    untreated_audio_paths = get_new_audio_filepaths(list(audio_file_paths.rglob("*.wav")))
    return untreated_audio_paths


def get_new_audio_filepaths(paths):
    # Check if files have already been treated in local db,
    # if so remove entry from paths list
    df_db = pd.read_csv(local_db, sep=',', encoding='utf-8')
    new_paths = []
    for path in paths:
        if str(path) not in df_db['OPath'].values:
            new_paths.append(path)
            #paths.remove(path)
    return new_paths
    #return paths


def treat_audio(paths, output_path, mode):
    # Track counter
    tcounter = 0
    # Create today's directory
    current_timestamp = datetime.now()
    new_output_path = os.path.join(output_path, current_timestamp.strftime("%Y_%m_%d_%H_%M"))
    dest_path = new_output_path
    if not os.path.isdir(new_output_path):
        os.mkdir(new_output_path)
    # Treat path by path
    artists =[]
    songs = []
    for audio_file in paths:
        # create output directories if it doesn't exist
        afilepath, copy_path, artists, songs = create_copy_directories(audio_file, new_output_path, artists, songs)
        #print(audio_file)
        if mode == "denoise":
            read_audio_meta(afilepath, copy_path)  # Treat data denoise
            tcounter = tcounter + 1
        else:
            convert_to_mp3(afilepath, copy_path)   # Treat data convert
            tcounter = tcounter + 1

    return tcounter, dest_path, artists, songs


def create_copy_directories(afilepath, outputp, artists, songs):
    pathinfo = afilepath.parts
    copy_path = os.path.join(outputp, pathinfo[7], pathinfo[8], pathinfo[9])
    # Collect job details
    if pathinfo[8] not in artists:
        artists.append(pathinfo[8])
    if pathinfo[9] not in songs:
        songs.append(pathinfo[9])
    if not os.path.isdir(copy_path):
        os.makedirs(copy_path)  # create path
        #read_audio_meta(afilepath, copy_path)  # Treat data denoise test 1 file

    return afilepath, copy_path, artists, songs


def convert_to_mp3(afpath, copypath):
    new_a_file_name = pathlib.Path(afpath).stem + ".mp3"
    current_wav_file = pydub.AudioSegment.from_wav(afpath)
    current_wav_file.export(os.path.join(copypath, new_a_file_name), format="mp3")

    # Get metadata as dict
    audioinfo = {header[0]: afpath, header[1]: new_a_file_name, header[2]: current_wav_file.dBFS,
                 header[3]: current_wav_file.channels, header[4]: current_wav_file.sample_width,
                 header[5]: current_wav_file.frame_rate, header[6]: current_wav_file.rms,
                 header[7]: current_wav_file.max_dBFS, header[8]: current_wav_file.duration_seconds,
                 header[9]: os.path.join(copypath, new_a_file_name), header[10]: "yes"}

    # Update csv file to yes for treated
    # Add csv entry
    add_csv_entry(audioinfo)


def read_audio_meta(audiop, copyp):
    # Read metadata of audio file and save to csv
    current_wav_file = pydub.AudioSegment.from_file(file=audiop, format="wav")
    a_file_name = os.path.basename(audiop)
    spectrograms = "spectrograms"

    # Create and save spectogram of original file
    specsp = os.path.join(copyp, spectrograms)
    if not os.path.isdir(specsp):
        # create path !!!
        os.makedirs(specsp)
        #print(specsp)
    imgpname = a_file_name.split('.')[0] + ".png"
    save_mel_spectogram(audiop, imgpname, specsp)

    # Denoize current sound and Get noise profile of song first via find silence algorithm
    # if no noise profile was created run denoise without optional argument
    noise_path = find_silence(audiop)
    if noise_path is not None:
        sr_noize, y_noize = wavfile.read(noise_path)
        sr_current, y = wavfile.read(audiop)
        clean_signal = nr.reduce_noise(y_noise=y_noize, y=y, sr=sr_current,
                                       n_std_thresh_stationary=1.5, stationary=True)
    else:
        sr_current, y = wavfile.read(audiop)
        clean_signal = nr.reduce_noise(y=y, sr=sr_current,
                                       n_std_thresh_stationary=1.5, stationary=True)

    # Save to copy path
    new_a_filepath = os.path.join(copyp, a_file_name)
    wavfile.write(filename=new_a_filepath, rate=sr_current, data=clean_signal)

    # Create and save mel spectogram of new noise reduced file
    imgpname_treated = a_file_name.split('.')[0] + "_treated.png"
    save_mel_spectogram(new_a_filepath, imgpname_treated, specsp)

    # Get metadata as dict
    audioinfo = {header[0]: audiop, header[1]: a_file_name, header[2]: current_wav_file.dBFS,
                 header[3]: current_wav_file.channels, header[4]: current_wav_file.sample_width,
                 header[5]: current_wav_file.frame_rate, header[6]: current_wav_file.rms,
                 header[7]: current_wav_file.max_dBFS, header[8]: current_wav_file.duration_seconds,
                 header[9]: os.path.join(copyp, a_file_name), header[10]: "yes"}

    # Update csv file to yes for treated
    # Add csv entry
    add_csv_entry(audioinfo)


def save_mel_spectogram(audiopath, filename, cpypath):
    y, sr = librosa.load(audiopath)
    y = y[:100000] # shorten audio for spectogram
    window_size = 1024
    window = np.hanning(window_size)
    stft = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    axes = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=axes, y_axis='log', x_axis='time')
    fig.savefig(os.path.join(cpypath, filename))

    #return sr


def add_csv_entry(entrydict):
    with open(local_db, 'a', newline="") as file:
        csvwriter = csv.DictWriter(file, fieldnames=header)
        csvwriter.writerow(entrydict)


def update_csv_entry(apath):
    df = pd.read_csv("localdb.csv")
    df.loc[apath, 'Treated'] = 'yes'

    # writing into the file
    df.to_csv("localdb.csv", index=False)


def find_silence(input_filep):
    soundclip = pydub.AudioSegment.from_file(input_filep, format="wav")
    #y, sr = wavfile.read(input_filep)
    filename = os.path.basename(input_filep)
    silence_file = os.path.join("tempData", filename)
    ms = 0
    current_silence = 0
    longest_time = 500
    longest_value = None
    #bitrate = str((soundclip.frame_rate * soundclip.frame_width * 8 * soundclip.channels) / 1000)

    for i in soundclip:
        if i.dBFS > -38.0:
            length = ms - current_silence
            if length > longest_time:
                longest_value = soundclip[current_silence : ms]
                longest_time = length
            current_silence = ms + 1
        ms += 1

    if longest_value is not None:
        print("Longest segment " + str(longest_time/1000.0) + " seconds")
        clip = longest_value[200:-200]
        # Avoid saving noise clips longer than 5s
        if clip.duration_seconds > 5:
            print("Shortening")
            clip = clip[:4000]  # If file is too big take last 4 seconds by default
        clip.export(silence_file, format="wav")
        return silence_file
    return None


def send_job_done_mail(jobid):
    port = 465  # For SSL
    password = "RkE7C6wP"
    smtp_server = "smtp.gmail.com"
    current_timestamp = datetime.now()

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("56955@etu.he2b.be", password)

    sender_email = "56955@etu.he2b.be"
    receiver_email = "jean.rene.nshuti@gmail.com"

    message = f'Subject: Python Job completed. \n\n' \
              f'Your python job with job ID: {jobid} was completed at {current_timestamp}.'

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def clean_temp_data():
    for f in os.listdir("tempData"):
        os.remove(os.path.join("tempData", f))


"""
    This section of code handles all text summary functions
"""


# Summary Job details
class SummaryJob:
    def __init__(self):
        self.authors = ""
        self.title = ""
        self.mode = ""
        self.dest = r''


def create_initial_summary_table():
    # Check if table exists, if not, create table
    dbconn = connect_to_db()
    db_cursor = dbconn.cursor()

    db_cursor.execute("SHOW TABLES")
    if "job" not in db_cursor.fetchall():
        sql_query = "CREATE TABLE IF NOT EXISTS SUMMARY_JOBS" \
                    "(JOBID INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY," \
                    "TIME_STAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP  ," \
                    "MODE VARCHAR(20) NOT NULL  ," \
                    "SOURCE_PATH VARCHAR(255) NOT NULL  ," \
                    "DEST_PATH VARCHAR(255) NOT NULL  ," \
                    "AUTHOR VARCHAR(128) NULL  ," \
                    "ARTICLE_TITLE VARCHAR(255) NULL) "
        db_cursor.execute(sql_query.lower())

    dbconn.close()


def create_copy_directory():
    # Create new copy directory
    current_timestamp = datetime.now()
    new_output_path = os.path.join(os.getcwd(), "out_" + current_timestamp.strftime("%Y_%m_%d_%H_%M") + "_summary")

    if not os.path.isdir(new_output_path):
        os.makedirs(new_output_path )  # create path

    return new_output_path


def check_db_table():
    # Make sure db and table have been created
    initial_mysql_connect()
    create_initial_summary_table()


def treat_pdf(article_path):
    page_text = []

    check_db_table()
    reader = PdfReader(article_path)
    meta = reader.metadata
    for page in reader.pages:
        page_text.append(page.extractText())

    # Collect job details
    fulltext = '\n'.join(page_text)
    author = meta.author
    title = meta.title
    cpath = create_copy_directory()

    return fulltext, title, author, cpath


def treat_online_article(article_url):
    check_db_table()

    article = Article(article_url)
    article.download()
    article.parse()
    # Collect job details
    authors = article.authors
    title = article.title
    text = article.text
    cpath = create_copy_directory()

    return text, title, authors, cpath


# Algorithm provided by Turing for Developers
# @https://www.turing.com/kb/5-powerful-text-summarization-techniques-in-python
def extractive_summary(text, linepercentage):
    # Tokenize the text
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Create a frequency table to keep a score of each word
    freq_table = dict()
    for word in words:
        word = word.lower()
        if word in stop_words:
            continue
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    # Create a dictionary to keep the score of each sentence
    sentences = sent_tokenize(text)
    sentence_value = dict()

    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                if word in sentence.lower():
                    if sentence in sentence_value:
                        sentence_value[sentence] += freq
                    else:
                        sentence_value[sentence] = freq

    # Define the average value from the original text
    sum_values = 0
    for sentence in sentence_value:
        sum_values += sentence_value[sentence]
    average = int(sum_values / len(sentence_value))

    # store the sentences into summary
    summary = ''

    for sentence in sentences:
        if (sentence in sentence_value) and (sentence_value[sentence] > (linepercentage * average)):
            summary += "\n" + sentence

    return summary


# Algorithm provided by Dante Sblendorio
# @https://www.activestate.com/blog/how-to-do-text-summarization-with-python/
def abstractive_summary(text, linepercentage):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc]
    word_frequencies = {}

    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}

    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens) * linepercentage)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = '\n'.join(final_summary)

    return summary


def output_summary(summary, outputp, author, title):
    summary_file = os.path.join(outputp, "output" + ".txt")
    print(summary_file)
    # Create file doesn't exist and write to it
    with open(summary_file, 'w+', encoding='utf-8') as f:
        f.write(title+"\n")
        f.write(author + "\n")
        f.write(summary)
        f.close()


def register_new_summary_job(mode, srcpath, destpath, authors, title):
    dbconn = connect_to_db()
    db_cursor = dbconn.cursor()

    sql_query = "INSERT INTO summary_jobs (mode, source_path, dest_path, author, article_title)" \
                " VALUES (%s, %s, %s, %s, %s)"
    row_values = (mode, srcpath, destpath, ''.join(authors), str(title))
    db_cursor.execute(sql_query, row_values)

    dbconn.commit()

    if db_cursor.rowcount > 0:
        print("Successfully persisted row of data")
    else:
        print("Failed to insert to db")

    return db_cursor.lastrowid
