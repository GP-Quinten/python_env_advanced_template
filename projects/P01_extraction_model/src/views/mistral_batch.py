import streamlit as st
import time
from mistralai import Mistral
import os

def load_jobs(status):
    # Simulated data loading for demonstration purposes
    # Replace with your actual API calls
    mistral = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
    response = mistral.batch.jobs.list(status=status).model_dump()
    return response

def cancel_job(job_id):
    mistral = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
    canceled_job = mistral.batch.jobs.cancel(job_id=job_id)
    return canceled_job

def cancel_all_jobs(jobs):
    for job in jobs:
        cancel_job(job['id'])

def display_jobs_compact(jobs, status):
    st.subheader(f"{status} Jobs | Total: {len(jobs)}")
    for job in jobs:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        with col1:
            st.write(f"**ID**: {job['id']}")
            st.write(f"**Model**: {job['model']}")
        with col2:
            st.write(f"**Status**: {job['status']}")
            if status == "RUNNING":
                st.write(f"**Progress**: {job['completed_requests']}/{job['total_requests']}")
                progress = job['completed_requests'] / job['total_requests'] * 100
                st.progress(progress / 100)
        with col3:
            if 'started_at' in job:
                st.write(f"**Started**: {time.ctime(job['started_at'])}")
            if 'completed_at' in job and job['completed_at']:
                duration = job['completed_at'] - job['started_at']
                st.write(f"**Duration**: {duration}s")
        with col4:
            if status in ["QUEUED", "RUNNING"]:
                if st.button("Cancel", key=f"cancel-{job['id']}"):
                    cancel_job(job['id'])
                    st.experimental_rerun()
        st.write("---")

def main():
    st.title("Batch Jobs Dashboard")

    mistral = Mistral(api_key=os.environ['MISTRAL_API_KEY'])

    if st.button("Cancel All QUEUED and RUNNING Jobs"):
        for status in ["QUEUED", "RUNNING"]:
            jobs_data = load_jobs(status=status)
            if jobs_data['total'] > 0:
                cancel_all_jobs(jobs_data['data'])
        st.experimental_rerun()

    if st.button("Refresh Jobs"):
        st.experimental_rerun()

    statuses = ["RUNNING", "QUEUED", "FAILED", "TIMEOUT_EXCEEDED", "SUCCESS", "CANCELLED"]

    for status in statuses:
        jobs_data = load_jobs(status=status)
        if jobs_data['total'] > 0:
            display_jobs_compact(jobs_data['data'], status)

if __name__ == "__main__":
    main()
