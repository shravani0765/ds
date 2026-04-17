import { DataPoint } from './types';

// Representative subset of the Indian IT Jobs dataset with numerical salaries (LPA)
export const ADULT_DATASET: DataPoint[] = [
  { age: 24, gender: 'Male', education: 'B.Tech', occupation: 'Software Engineer', location: 'Bengaluru', workHours: 45, experience: 2, salary: 8.5 },
  { age: 32, gender: 'Male', education: 'M.Tech', occupation: 'DevOps Engineer', location: 'Hyderabad', workHours: 50, experience: 8, salary: 24.0 },
  { age: 28, gender: 'Female', education: 'B.E', occupation: 'Frontend Developer', location: 'Pune', workHours: 40, experience: 5, salary: 12.5 },
  { age: 35, gender: 'Female', education: 'MBA', occupation: 'Product Manager', location: 'Bengaluru', workHours: 45, experience: 10, salary: 32.0 },
  { age: 22, gender: 'Male', education: 'BCA', occupation: 'QA Engineer', location: 'Chennai', workHours: 40, experience: 1, salary: 4.5 },
  { age: 40, gender: 'Male', education: 'PhD', occupation: 'Data Scientist', location: 'Gurugram', workHours: 45, experience: 15, salary: 45.0 },
  { age: 29, gender: 'Female', education: 'B.Tech', occupation: 'Backend Developer', location: 'Hyderabad', workHours: 40, experience: 6, salary: 18.0 },
  { age: 31, gender: 'Male', education: 'B.E', occupation: 'Cloud Architect', location: 'Bengaluru', workHours: 50, experience: 9, salary: 28.5 },
  { age: 26, gender: 'Female', education: 'M.Tech', occupation: 'Fullstack Developer', location: 'Noida', workHours: 40, experience: 3, salary: 9.0 },
  { age: 38, gender: 'Male', education: 'B.Tech', occupation: 'Engineering Manager', location: 'Mumbai', workHours: 55, experience: 14, salary: 55.0 },
];

export const generateSyntheticData = (count: number): DataPoint[] => {
  const data: DataPoint[] = [...ADULT_DATASET];
  const educations = ['B.Tech', 'M.Tech', 'B.E', 'MBA', 'PhD', 'BCA', 'MCA', 'B.Sc IT'];
  const occupations = ['Software Engineer', 'DevOps Engineer', 'Frontend Developer', 'Backend Developer', 'Fullstack Developer', 'Data Scientist', 'QA Engineer', 'Cloud Architect', 'Product Manager', 'Engineering Manager', 'Security Engineer'];
  const locations = ['Bengaluru', 'Hyderabad', 'Pune', 'Chennai', 'Mumbai', 'Noida', 'Gurugram', 'Kolkata'];
  
  for (let i = 0; i < count; i++) {
    const gender = Math.random() > 0.4 ? 'Male' : 'Female';
    const age = Math.floor(Math.random() * 30) + 21;
    const education = educations[Math.floor(Math.random() * educations.length)];
    const occupation = occupations[Math.floor(Math.random() * occupations.length)];
    const location = locations[Math.floor(Math.random() * locations.length)];
    const experience = Math.max(0, age - 22);
    const workHours = Math.floor(Math.random() * 20) + 35;
    
    // Logic for approximate salary (LPA)
    let baseSalary = 4.0; // Base for freshers
    baseSalary += experience * 2.5; // Experience increment
    
    if (education === 'M.Tech' || education === 'MBA') baseSalary += 3.0;
    if (education === 'PhD') baseSalary += 7.0;
    
    if (occupation === 'Engineering Manager' || occupation === 'Cloud Architect') baseSalary += 15.0;
    if (occupation === 'Data Scientist' || occupation === 'Security Engineer') baseSalary += 8.0;
    
    if (location === 'Bengaluru' || location === 'Hyderabad' || location === 'Mumbai') baseSalary += 4.0;
    
    // Intentional bias for detection: 10% lower for females on average
    if (gender === 'Female') baseSalary *= 0.9;
    
    // Add some noise (+/- 15%)
    const noise = 0.85 + Math.random() * 0.3;
    const salary = parseFloat((baseSalary * noise).toFixed(1));
    
    data.push({ age, gender, education, occupation, location, workHours, experience, salary });
  }
  
  return data;
};
