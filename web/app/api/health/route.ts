import { NextResponse } from "next/server";

const apiBase = process.env.PRISM_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${apiBase}/ready`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    return NextResponse.json(
      { ready: response.ok, status: response.status },
      { status: response.ok ? 200 : 503 },
    );
  } catch {
    return NextResponse.json({ ready: false, status: 503 }, { status: 503 });
  }
}
