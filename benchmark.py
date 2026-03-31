import asyncio
from client import send_request
from utils import compute_stats

TEXT = "Hello this is a test for TTS serving system. This benchmark measures real TTS inference performance."

async def run_concurrent(n):
    tasks = [send_request(TEXT) for _ in range(n)]
    results = await asyncio.gather(*tasks)

    ttfts = [r[0] for r in results]
    rtfs = [r[1] for r in results]

    return ttfts, rtfs

async def main():
    for c in [1, 5, 10]:
        print(f"\n=== Concurrency {c} ===")

        # warmup
        for _ in range(2):
            await run_concurrent(1)

        all_ttft = []
        all_rtf = []

        for _ in range(3):
            ttfts, rtfs = await run_concurrent(c)
            all_ttft.extend(ttfts)
            all_rtf.extend(rtfs)

        print({
            "ttft": compute_stats(all_ttft),
            "rtf": compute_stats(all_rtf)
        })

if __name__ == "__main__":
    asyncio.run(main())