import os
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Resume Analyzer & ATS Score Checker",
    page_icon="📄",
    layout="wide"
)

class ATSAnalysis(BaseModel):
    match_score: int = Field(description="ATS match percentage score from 0 to 100")
    matching_skills: list[str] = Field(description="List of skills and qualifications found in both resume and job description")
    missing_keywords: list[str] = Field(description="Important skills, tools, or keywords from the job description missing in the resume")
    improvements: list[str] = Field(description="Exactly 5 specific, actionable suggestions to improve the resume for this role")
    summary_verdict: str = Field(description="A concise 2-sentence overall evaluation of the candidate's fit")

def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def analyze_resume(resume_text: str, job_description: str, api_key: str) -> ATSAnalysis:
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert Applicant Tracking System (ATS) and Technical Recruiter.
    Evaluate the following candidate resume against the provided job description.
    
    Be objective and strict:
    1. Calculate a realistic ATS match score out of 100 based on skill overlap, experience relevance, and keyword alignment.
    2. Extract the exact skills/technologies present in both the resume and job description.
    3. Identify critical missing keywords, tools, or qualifications required by the job description.
    4. Provide exactly 5 concrete, actionable improvements tailored to this specific resume and job description.
    
    --- RESUME TEXT ---
    {resume_text}
    
    --- JOB DESCRIPTION ---
    {job_description}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ATSAnalysis,
            temperature=0.2,
        ),
    )
    return response.parsed

st.title("📄 AI Resume Analyzer & ATS Score Checker")
st.markdown("Upload your resume (PDF) and paste the target Job Description to check your ATS match score and get actionable optimization tips.")

with st.sidebar:
    st.header("⚙️ Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "")
    if not env_key:
        try:
            env_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            env_key = ""
        
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="Loaded automatically from .env or Streamlit Secrets if configured."
    )
    st.info("Built with Python, Streamlit, pypdf & Google Gemini API.")

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("1. Upload Resume (PDF)", type=["pdf"])

with col2:
    job_description = st.text_area(
        "2. Paste Job Description",
        height=200,
        placeholder="Paste the full job role requirements, responsibilities, and preferred skills here..."
    )

analyze_btn = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

if analyze_btn:
    if not api_key:
        st.error("Please provide your Gemini API Key in the sidebar or .env file.")
    elif not uploaded_resume:
        st.warning("Please upload your resume in PDF format.")
    elif not job_description.strip():
        st.warning("Please paste a job description to compare against.")
    else:
        with st.spinner("Extracting resume text and analyzing against ATS criteria..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_resume)
                if not resume_text.strip():
                    st.error("Could not extract text from this PDF. Please upload a text-based PDF (not a scanned image).")
                else:
                    result = analyze_resume(resume_text, job_description, api_key)
                    
                    st.divider()
                    st.subheader("📊 ATS Evaluation Results")
                    
                    score_col, verdict_col = st.columns([1, 3])
                    with score_col:
                        st.metric(label="ATS Match Score", value=f"{result.match_score} / 100")
                        st.progress(min(max(result.match_score, 0), 100) / 100)
                    with verdict_col:
                        st.markdown("**Recruiter Verdict:**")
                        st.write(result.summary_verdict)
                    
                    st.divider()
                    match_col, miss_col = st.columns(2)
                    
                    with match_col:
                        st.subheader("✅ Matching Skills Found")
                        if result.matching_skills:
                            for skill in result.matching_skills:
                                st.markdown(f"- **{skill}**")
                        else:
                            st.write("No direct skill matches detected.")
                            
                    with miss_col:
                        st.subheader("⚠️ Missing Keywords & Skills")
                        if result.missing_keywords:
                            for keyword in result.missing_keywords:
                                st.markdown(f"- `{keyword}`")
                        else:
                            st.write("Great job! No major keywords missing.")
                    
                    st.divider()
                    st.subheader("💡 5 Actionable Improvements")
                    for idx, tip in enumerate(result.improvements, start=1):
                        st.markdown(f"**{idx}.** {tip}")
                        
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")