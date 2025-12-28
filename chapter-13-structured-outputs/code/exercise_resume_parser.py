"""
Exercise Solution: Resume Parser

This module implements a structured resume parser that extracts
information from resume text into validated Pydantic models.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
import re
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# Define nested schemas for resume structure

class ContactInfo(BaseModel):
    """Contact information from a resume."""
    
    name: str = Field(description="Full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="City, State or location")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn URL or username")
    website: Optional[str] = Field(default=None, description="Personal website URL")


class WorkExperience(BaseModel):
    """A single work experience entry."""
    
    company: str = Field(description="Company or organization name")
    title: str = Field(description="Job title")
    start_date: Optional[str] = Field(
        default=None, 
        description="Start date (flexible format: 'Jan 2020', '2020', etc.)"
    )
    end_date: Optional[str] = Field(
        default=None, 
        description="End date or 'Present' if current role"
    )
    description: Optional[str] = Field(
        default=None, 
        description="Brief description of responsibilities"
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Key achievements or responsibilities"
    )


class Education(BaseModel):
    """A single education entry."""
    
    institution: str = Field(description="School or university name")
    degree: Optional[str] = Field(
        default=None, 
        description="Degree type (BS, MS, PhD, etc.)"
    )
    field: Optional[str] = Field(
        default=None, 
        description="Field of study or major"
    )
    graduation_year: Optional[str] = Field(
        default=None, 
        description="Graduation year or expected graduation"
    )
    gpa: Optional[str] = Field(
        default=None, 
        description="GPA if mentioned"
    )
    honors: list[str] = Field(
        default_factory=list,
        description="Honors, awards, or notable achievements"
    )


class Skills(BaseModel):
    """Categorized skills from a resume."""
    
    technical: list[str] = Field(
        default_factory=list,
        description="Technical skills (programming, tools, technologies)"
    )
    soft: list[str] = Field(
        default_factory=list,
        description="Soft skills (communication, leadership, etc.)"
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Human languages spoken"
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Professional certifications"
    )


class ResumeData(BaseModel):
    """Complete structured resume data."""
    
    contact: ContactInfo = Field(description="Contact information")
    summary: Optional[str] = Field(
        default=None,
        description="Professional summary or objective"
    )
    experience: list[WorkExperience] = Field(
        default_factory=list,
        description="Work experience entries (most recent first)"
    )
    education: list[Education] = Field(
        default_factory=list,
        description="Education entries"
    )
    skills: Skills = Field(
        default_factory=Skills,
        description="Categorized skills"
    )
    projects: list[str] = Field(
        default_factory=list,
        description="Notable projects mentioned"
    )
    volunteer: list[str] = Field(
        default_factory=list,
        description="Volunteer experience or community involvement"
    )


class ResumeParser:
    """
    Parses resume text into structured data.
    
    This class extracts information from resume text and validates
    it against a comprehensive schema with retry logic for robustness.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the parser.
        
        Args:
            verbose: Whether to print debug information
        """
        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"
        self.verbose = verbose
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ResumeParser] {message}")
    
    def _clean_response(self, text: str) -> str:
        """Clean potential markdown formatting from response."""
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1:
            return text[start:end + 1]
        
        return text
    
    def _build_prompt(self, resume_text: str) -> str:
        """Build the extraction prompt."""
        return f"""Parse this resume and extract structured information.

Return a JSON object with this structure:
{{
    "contact": {{
        "name": "string (required)",
        "email": "string or null",
        "phone": "string or null",
        "location": "string or null",
        "linkedin": "string or null",
        "website": "string or null"
    }},
    "summary": "string or null (professional summary/objective)",
    "experience": [
        {{
            "company": "string (required)",
            "title": "string (required)",
            "start_date": "string or null (e.g., 'Jan 2020', '2020')",
            "end_date": "string or null (e.g., 'Dec 2022', 'Present')",
            "description": "string or null",
            "highlights": ["array of strings"]
        }}
    ],
    "education": [
        {{
            "institution": "string (required)",
            "degree": "string or null (BS, MS, etc.)",
            "field": "string or null (major/field of study)",
            "graduation_year": "string or null",
            "gpa": "string or null",
            "honors": ["array of strings"]
        }}
    ],
    "skills": {{
        "technical": ["programming languages, tools, frameworks"],
        "soft": ["leadership, communication, etc."],
        "languages": ["human languages spoken"],
        "certifications": ["professional certifications"]
    }},
    "projects": ["notable projects mentioned"],
    "volunteer": ["volunteer experience"]
}}

Resume:
---
{resume_text}
---

Return ONLY the JSON object, no explanations or markdown."""

    def parse(self, resume_text: str, max_retries: int = 2) -> ResumeData:
        """
        Parse resume text into structured data.
        
        Args:
            resume_text: The resume text to parse
            max_retries: Number of retry attempts for parsing errors
            
        Returns:
            ResumeData object with extracted information
            
        Raises:
            ValueError: If parsing fails after all retries
        """
        prompt = self._build_prompt(resume_text)
        self._log(f"Parsing resume ({len(resume_text)} chars)")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system="You are a resume parsing assistant. Extract all available information and respond only with valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        self._log(f"Received response ({len(response_text)} chars)")
        last_error = None
        
        for attempt in range(max_retries + 1):
            self._log(f"Parse attempt {attempt + 1}/{max_retries + 1}")
            
            try:
                cleaned = self._clean_response(response_text)
                data = json.loads(cleaned)
                result = ResumeData.model_validate(data)
                self._log("Successfully parsed and validated!")
                return result
                
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e.msg}"
                self._log(f"JSON error: {last_error}")
            except ValidationError as e:
                errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                last_error = f"Validation failed: {'; '.join(errors[:3])}"
                self._log(f"Validation error: {last_error}")
            
            if attempt < max_retries:
                self._log("Requesting corrected response...")
                correction = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system="You are a resume parsing assistant. Respond only with valid JSON.",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user", 
                            "content": f"Error: {last_error}\n\nPlease return corrected JSON only."
                        }
                    ]
                )
                response_text = correction.content[0].text
        
        raise ValueError(f"Failed to parse resume: {last_error}")


def print_resume(data: ResumeData) -> None:
    """Pretty print parsed resume data."""
    print("\n" + "=" * 60)
    print("PARSED RESUME")
    print("=" * 60)
    
    # Contact info
    print(f"\n📧 CONTACT")
    print(f"   Name: {data.contact.name}")
    if data.contact.email:
        print(f"   Email: {data.contact.email}")
    if data.contact.phone:
        print(f"   Phone: {data.contact.phone}")
    if data.contact.location:
        print(f"   Location: {data.contact.location}")
    if data.contact.linkedin:
        print(f"   LinkedIn: {data.contact.linkedin}")
    
    # Summary
    if data.summary:
        print(f"\n📝 SUMMARY")
        print(f"   {data.summary}")
    
    # Experience
    if data.experience:
        print(f"\n💼 EXPERIENCE ({len(data.experience)} positions)")
        for exp in data.experience:
            date_range = ""
            if exp.start_date:
                date_range = f" ({exp.start_date}"
                if exp.end_date:
                    date_range += f" - {exp.end_date})"
                else:
                    date_range += ")"
            
            print(f"\n   {exp.title} at {exp.company}{date_range}")
            if exp.description:
                print(f"   {exp.description}")
            for highlight in exp.highlights[:3]:  # Show first 3
                print(f"   • {highlight}")
    
    # Education
    if data.education:
        print(f"\n🎓 EDUCATION ({len(data.education)} entries)")
        for edu in data.education:
            degree_str = ""
            if edu.degree:
                degree_str = edu.degree
                if edu.field:
                    degree_str += f" in {edu.field}"
            
            print(f"\n   {edu.institution}")
            if degree_str:
                print(f"   {degree_str}")
            if edu.graduation_year:
                print(f"   Graduated: {edu.graduation_year}")
            if edu.gpa:
                print(f"   GPA: {edu.gpa}")
    
    # Skills
    print(f"\n🛠️ SKILLS")
    if data.skills.technical:
        print(f"   Technical: {', '.join(data.skills.technical[:10])}")
    if data.skills.soft:
        print(f"   Soft: {', '.join(data.skills.soft[:5])}")
    if data.skills.languages:
        print(f"   Languages: {', '.join(data.skills.languages)}")
    if data.skills.certifications:
        print(f"   Certifications: {', '.join(data.skills.certifications)}")
    
    # Projects
    if data.projects:
        print(f"\n🚀 PROJECTS")
        for project in data.projects[:3]:
            print(f"   • {project}")


if __name__ == "__main__":
    parser = ResumeParser(verbose=True)
    
    # Test resume 1: Software Engineer
    resume1 = """
JANE DOE
Software Engineer
jane.doe@email.com | (555) 123-4567 | San Francisco, CA
linkedin.com/in/janedoe | github.com/janedoe

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years of expertise in full-stack development,
specializing in Python, React, and cloud technologies. Passionate about building
scalable systems and mentoring junior developers.

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | March 2022 - Present
- Led development of microservices architecture serving 1M+ daily users
- Reduced API latency by 40% through optimization and caching strategies
- Mentored team of 3 junior developers
- Technologies: Python, FastAPI, PostgreSQL, Redis, AWS

Software Engineer | StartupXYZ | June 2019 - February 2022
- Built React frontend for customer dashboard
- Implemented CI/CD pipeline reducing deployment time by 60%
- Developed RESTful APIs for mobile applications
- Technologies: React, Node.js, MongoDB, Docker

Junior Developer | WebAgency | January 2018 - May 2019
- Maintained and enhanced client websites
- Collaborated with design team on UI implementations

EDUCATION

M.S. Computer Science | Stanford University | 2019
- Focus: Distributed Systems
- GPA: 3.8

B.S. Computer Science | UC Berkeley | 2017
- Dean's List, Magna Cum Laude

SKILLS
Technical: Python, JavaScript, TypeScript, React, Node.js, FastAPI, PostgreSQL,
MongoDB, Redis, Docker, Kubernetes, AWS, GCP, Git, CI/CD
Soft Skills: Leadership, Mentoring, Communication, Problem Solving
Languages: English (Native), Spanish (Conversational)
Certifications: AWS Solutions Architect Associate, Kubernetes Administrator

PROJECTS
- Open source contributor to FastAPI framework
- Personal finance app with 10K+ downloads
    """
    
    print("=" * 60)
    print("TEST 1: Software Engineer Resume")
    print("=" * 60)
    
    try:
        result1 = parser.parse(resume1)
        print_resume(result1)
    except ValueError as e:
        print(f"Failed: {e}")
    
    # Test resume 2: Marketing Professional
    resume2 = """
MICHAEL CHEN
Marketing Manager

Contact: michael.chen@gmail.com
Phone: 415-555-9876
Location: New York, NY

ABOUT ME
Creative marketing professional with 7 years of experience driving brand growth
through innovative digital campaigns. Expert in social media strategy, content
marketing, and data analytics.

WORK HISTORY

Marketing Manager, Global Brands Co. (2021-Present)
Oversee $2M annual marketing budget across digital channels.
- Increased social media engagement by 150% YoY
- Launched influencer partnership program generating $500K revenue
- Led rebrand initiative increasing brand recognition by 35%

Senior Marketing Specialist, Digital First Agency (2018-2021)
Managed accounts for Fortune 500 clients.
Key wins: Launched viral campaign with 10M impressions

Marketing Coordinator, Local Business Inc. (2016-2018)
Entry-level role managing social media and email campaigns.

EDUCATION
MBA, Marketing Concentration - NYU Stern School of Business, 2020
BA Communications - Boston University, 2016

EXPERTISE
- Digital Marketing: SEO, SEM, PPC, Social Media, Email Marketing
- Tools: Google Analytics, Salesforce, HubSpot, Hootsuite, Adobe Creative Suite
- Soft Skills: Strategic Thinking, Team Leadership, Client Relations
- Languages: English, Mandarin Chinese

CERTIFICATIONS
- Google Analytics Certified
- HubSpot Inbound Marketing Certified
- Facebook Blueprint Certified

VOLUNTEER
- Marketing advisor for local nonprofit (2019-present)
- Career mentor at alma mater
    """
    
    print("\n" + "=" * 60)
    print("TEST 2: Marketing Manager Resume")
    print("=" * 60)
    
    try:
        result2 = parser.parse(resume2)
        print_resume(result2)
    except ValueError as e:
        print(f"Failed: {e}")
    
    # Show JSON output for one resume
    print("\n" + "=" * 60)
    print("JSON OUTPUT EXAMPLE")
    print("=" * 60)
    
    if result1:
        # Convert to dict and show truncated JSON
        json_output = result1.model_dump()
        print(json.dumps(json_output, indent=2)[:1500] + "\n... (truncated)")
