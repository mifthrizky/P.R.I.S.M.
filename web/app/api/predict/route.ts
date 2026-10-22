import { NextRequest, NextResponse } from "next/server";

const apiBase = process.env.PRISM_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const response = await fetch(`${apiBase}/v1/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(45_000),
    });
    const payload = await response.json();
    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        detail: "PRISM API is unavailable.",
        error: error instanceof Error ? error.name : "UnknownError",
      },
      { status: 503 },
    );
  }
}
