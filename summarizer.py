import argparse
import os
import pathlib

import LangageScriptLib

if __name__ == '__main__':

    # create command line parser
    parser = argparse.ArgumentParser()
    # add arguments to the parser
    parser.add_argument("article_path")
    parser.add_argument("summary_mode")
    args = parser.parse_args()

    if os.path.isfile(args.article_path) or args.article_path is not None:
        text = summary = ""
        jobid = None
        # Get extension of file
        ext = pathlib.Path(args.article_path).suffix
        print(ext)
        # Create job
        job = LangageScriptLib.SummaryJob()

        if ext == ".pdf":
            text, job.title, job.authors, job.dest = LangageScriptLib.treat_pdf(args.article_path)
        else:
            text, job.title, job.authors, job.dest = LangageScriptLib.treat_online_article(args.article_path)
        if args.summary_mode == "True":
            summary = LangageScriptLib.abstractive_summary(text, 0.9)
            job.mode = "abstractive"
        else:
            summary = LangageScriptLib.extractive_summary(text, 1.2)
            job.mode = "extractive"

        print(job.authors)
        print(job.title)
        LangageScriptLib.output_summary(summary, job.dest, job.authors, job.title)
        jobid = LangageScriptLib.register_new_summary_job(job.mode, args.article_path, job.dest, job.authors, job.title)
        LangageScriptLib.send_job_done_mail(jobid)

    else:
        print("did not recognise arguments")