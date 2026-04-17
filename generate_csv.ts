import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';
import { generateSyntheticData } from './dataset';

const outputDir = path.join(process.cwd(), 'public');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

const data = generateSyntheticData(5000); // 5000 rows

const csv = Papa.unparse(data);

const filePath = path.join(outputDir, 'indian_it_job_market.csv');
fs.writeFileSync(filePath, csv);
console.log(`Generated ${data.length} records in ${filePath}`);
