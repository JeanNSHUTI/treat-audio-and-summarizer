import argparse
import os
import LangageScriptLib
from pydub import playback

if __name__ == '__main__':
    #reduced_signal = LangageScriptLib.find_silence(r'C:\Users\Admin\Documents\HE2B\Master2\LangageScript\AudioTestData\ChanteurH\DifferentDAWFiles\backup, Rec (33).wav')

    # create command line parser
    parser = argparse.ArgumentParser()
    # add arguments to the parser
    parser.add_argument("audio_path")
    parser.add_argument("copy_mode")
    args = parser.parse_args()

    # Check if local csv database has been created, if not create it
    local_db_path = os.path.join(os.getcwd(), LangageScriptLib.local_db)
    if not os.path.isfile(local_db_path):
        LangageScriptLib.create_local_csv()

    if os.path.exists(args.audio_path):
        job = LangageScriptLib.AudioJob()
        job.set_mode(args.copy_mode)
        global_out_path = LangageScriptLib.create_global_output_path(args.audio_path)
        LangageScriptLib.print_hi(args.audio_path)
        found_files = LangageScriptLib.get_audio_filepaths(args.audio_path)
        if len(found_files) > 0:
            job.no_tracks_treated, job.destination_path, job.artists, job.songs = LangageScriptLib.treat_audio(found_files,
                                                                                                               global_out_path,
                                                                                                               job.mode)
            job_id = LangageScriptLib.register_new_job(job.no_tracks_treated, args.audio_path, job.destination_path,
                                                       job.mode, job.artists, job.songs)
            LangageScriptLib.send_job_done_mail(job_id)
            # Delete temporary data
            LangageScriptLib.clean_temp_data()
        else:
            print("No new files to be treated")
