
import { NextResponse } from 'next/server'

let history: any[] = [];

export async function GET() {
  return NextResponse.json(history);
}

export async function POST(request: Request) {
  const prediction = await request.json();
  prediction.id = history.length + 1;
  prediction.createdAt = new Date().toISOString();
  history.push(prediction);
  return NextResponse.json(prediction);
}
