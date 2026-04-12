import { GoogleGenAI, Type } from "@google/genai";

export interface ResumeData {
  age: number;
  gender: 'Male' | 'Female';
  education: string;
  occupation: string;
  location: string;
  workHours: number;
  experience: number;
  skills: string[];
}

export class ResumeService {
  private ai: GoogleGenAI;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }

  public async analyzeResume(text: string): Promise<ResumeData> {
    const response = await this.ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Analyze the following resume text and extract key professional details for an Indian IT job market context. 
      If some info is missing, provide a reasonable estimate based on the context.
      
      Resume Text:
      ${text}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            age: { type: Type.NUMBER, description: "Estimated age based on graduation or experience" },
            gender: { type: Type.STRING, enum: ["Male", "Female"], description: "Estimated gender if possible, default to 'Male' if unclear" },
            education: { type: Type.STRING, description: "Highest degree (e.g., B.Tech, M.Tech, MCA)" },
            occupation: { 
              type: Type.STRING, 
              enum: ["Software Engineer", "DevOps Engineer", "Frontend Developer", "Backend Developer", "Fullstack Developer", "Data Scientist", "QA Engineer", "Cloud Architect", "Product Manager", "Engineering Manager"],
              description: "Current or most relevant job role" 
            },
            location: {
              type: Type.STRING,
              enum: ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Mumbai", "Noida", "Gurugram", "Kolkata"],
              description: "Preferred or current location in India"
            },
            workHours: { type: Type.NUMBER, description: "Estimated weekly work hours based on role seniority, default to 45" },
            experience: { type: Type.NUMBER, description: "Total years of experience" },
            skills: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of technical skills" }
          },
          required: ["age", "gender", "education", "occupation", "location", "workHours", "experience", "skills"]
        }
      }
    });

    try {
      return JSON.parse(response.text || '{}') as ResumeData;
    } catch (e) {
      console.error("Failed to parse resume analysis", e);
      throw new Error("Could not analyze resume. Please try again.");
    }
  }
}
