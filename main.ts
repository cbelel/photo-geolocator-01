// photo-geolocator-01 entry: a GeoGuessr-style vision agent.
//
// For each of the instance's 25 street-level photos it asks a vision model
// (Claude Sonnet via the Anthropic gateway, falling back to GPT via the
// OpenAI gateway) where the photo was taken, and posts every answer to the
// env as soon as it is ready.
//
// Improvements over the kit starter:
//   * A real geolocation prompt: the model reads signs, language, architecture,
//     vegetation, road markings, driving side, plates and other clues, and
//     reports a confidence and a country alongside the coordinates.
//   * A second-chance call for photos whose first answer held no usable
//     coordinates (bounded, so the $1.00/trial model budget is respected).
//   * A country-centroid fallback: when the model names a country but gives no
//     coordinates, the country's centroid is posted instead of leaving the
//     photo unanswered (unanswered = 20,015 km, the worst possible error).
//   * Answers are posted incrementally, one photo at a time, so a trial that
//     ends early still keeps the answers it already found.
//   * Anthropic first, OpenAI fallback: whichever key the entrant pushed.
//
// Runs as the grader runs an entry, `node /bot/main.ts` on Node 22 (erasable
// TypeScript, the standard library only), with MAKEFUN_ENV_URL,
// ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY and OPENAI_BASE_URL / OPENAI_API_KEY.

const ANTHROPIC_MODEL = 'claude-sonnet-5-5';
const OPENAI_MODEL = 'gpt-6.1-sol';
const MAX_TOKENS = 1024;
const CONCURRENCY = 5;
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;

/** The geolocation prompt: make the model read the photo like a GeoGuessr player. */
const PROMPT =
  'You are an expert GeoGuessr player. Look at this street-level photo and work out where it was taken. ' +
  'Examine the clues carefully: the language and script on signs, shop names and road markings; the architecture and building styles; ' +
  'the vegetation, climate and landscape; which side of the road vehicles drive on; license plates; flags; the style of road signs, ' +
  'street furniture and power lines; any visible text, phone numbers or web addresses. ' +
  'Reason briefly about what country and region the clues point to, then narrow down to a city or area when you can. ' +
  'Reply with exactly one JSON object and nothing else: {"lat": <degrees north>, "lon": <degrees east>, "confidence": <0 to 1>, "country": "<country name or ISO code>"}. ' +
  'Give your best estimate even when unsure: a rough guess is far better than no guess.';

const SECOND_CHANCE_PROMPT =
  'Where was this photo taken? Look at the language on signs, the architecture, the vegetation and the road layout. ' +
  'Give your best estimate as one JSON object: {"lat": <degrees north>, "lon": <degrees east>, "country": "<country name>"}.';

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

interface ModelGuess {
  lat: number;
  lon: number;
  country?: string;
}

/** Country centroids (approximate), used only when the model names a country
 *  but gives no coordinates: a centroid guess beats the 20,015 km penalty
 *  of leaving a photo unanswered. */
const COUNTRY_CENTROIDS: Record<string, [number, number]> = {
  US: [39.8, -98.6], CA: [56.1, -106.3], MX: [23.6, -102.5], BR: [-14.2, -51.9],
  AR: [-38.4, -63.6], CL: [-35.7, -71.5], CO: [4.6, -74.1], PE: [-9.2, -75.0],
  GB: [54.0, -2.0], IE: [53.2, -8.2], FR: [46.6, 2.4], DE: [51.1, 10.4],
  ES: [40.3, -3.7], PT: [39.6, -8.0], IT: [42.8, 12.8], NL: [52.1, 5.3],
  BE: [50.5, 4.5], CH: [46.8, 8.2], AT: [47.6, 14.1], PL: [52.0, 19.4],
  CZ: [49.8, 15.5], SK: [48.7, 19.7], HU: [47.2, 19.5], RO: [45.9, 24.9],
  GR: [39.1, 22.0], TR: [38.9, 35.2], UA: [49.0, 32.0], RU: [61.5, 90.0],
  SE: [62.0, 15.0], NO: [64.5, 11.5], FI: [64.0, 26.0], DK: [56.0, 10.0],
  JP: [36.2, 138.3], KR: [36.4, 127.9], CN: [35.9, 104.2], IN: [21.1, 78.9],
  PK: [30.4, 69.3], BD: [23.7, 90.4], LK: [7.9, 80.8], TH: [15.9, 101.0],
  VN: [14.1, 108.3], MY: [4.2, 102.0], SG: [1.35, 103.8], ID: [-2.5, 118.0],
  PH: [12.9, 121.8], AU: [-25.3, 133.8], NZ: [-41.8, 172.8], PG: [-6.3, 147.0],
  ZA: [-29.0, 24.0], EG: [26.8, 30.8], NG: [9.1, 8.7], KE: [0.5, 37.9],
  TZ: [-6.4, 34.9], MA: [31.8, -6.1], DZ: [28.0, 1.7], ET: [9.1, 40.5],
  CI: [7.5, -5.5], GH: [7.9, -1.2], SN: [14.5, -14.5], IL: [31.0, 34.8],
  SA: [24.6, 45.1], AE: [24.3, 54.4], IQ: [33.0, 44.4], IR: [32.4, 53.7],
  KZ: [48.0, 66.9], UZ: [41.4, 64.6], MN: [46.9, 105.9], NP: [28.4, 84.1],
  MM: [21.9, 96.0], KH: [12.6, 105.0], LA: [19.9, 102.5], TW: [23.7, 121.0],
  HK: [22.3, 114.2], IS: [65.0, -18.7], EE: [58.6, 25.0], LV: [56.9, 24.6],
  LT: [55.2, 23.9], HR: [45.1, 15.2], RS: [44.0, 20.8], BG: [42.7, 25.5],
  SI: [46.1, 14.8], BA: [43.9, 17.7], AL: [41.2, 20.0], MK: [41.6, 21.7],
};

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = (process.env.ANTHROPIC_BASE_URL ?? '').replace(/\/+$/, '');
const openaiUrl = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const anthropicKey = process.env.ANTHROPIC_API_KEY ?? '';
const openaiKey = process.env.OPENAI_API_KEY ?? '';

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

const isLat = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The coordinates (and country, when given) an answer holds: its first JSON
 *  object with lat and lon, else its first two numbers in range. When the
 *  answer names a country but gives no coordinates, the country is still
 *  returned so the caller can fall back to its centroid. */
function parseGuess(text: string): ModelGuess | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      const c = o.country ?? o.country_code;
      const country = typeof c === 'string' ? countryCode(c) : undefined;
      if (isLat(lat) && isLon(lon)) return { lat, lon, country };
      if (country !== undefined) return { lat: NaN, lon: NaN, country };
    } catch {
      // Not JSON: keep looking.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  const country = mentionedCountry(text);
  if (country !== undefined) return { lat: NaN, lon: NaN, country };
  return null;
}

/** Full country names, matched as whole words: two-letter codes collide with
 *  ordinary English words ("it", "in", "us", "is"), so only full names are
 *  trusted in free text. */
const COUNTRY_NAMES: Record<string, string> = {
  'united states': 'US', usa: 'US', america: 'US', canada: 'CA', mexico: 'MX',
  brazil: 'BR', argentina: 'AR', chile: 'CL', colombia: 'CO', peru: 'PE',
  'united kingdom': 'GB', britain: 'GB', england: 'GB', scotland: 'GB', ireland: 'IE',
  france: 'FR', germany: 'DE', spain: 'ES', portugal: 'PT', italy: 'IT',
  netherlands: 'NL', belgium: 'BE', switzerland: 'CH', austria: 'AT', poland: 'PL',
  'czech republic': 'CZ', czechia: 'CZ', slovakia: 'SK', hungary: 'HU', romania: 'RO',
  greece: 'GR', turkey: 'TR', ukraine: 'UA', russia: 'RU', sweden: 'SE',
  norway: 'NO', finland: 'FI', denmark: 'DK', japan: 'JP', 'south korea': 'KR',
  korea: 'KR', china: 'CN', india: 'IN', pakistan: 'PK', bangladesh: 'BD',
  'sri lanka': 'LK', thailand: 'TH', vietnam: 'VN', malaysia: 'MY', singapore: 'SG',
  indonesia: 'ID', philippines: 'PH', australia: 'AU', 'new zealand': 'NZ',
  'papua new guinea': 'PG', 'south africa': 'ZA', egypt: 'EG', nigeria: 'NG',
  kenya: 'KE', tanzania: 'TZ', morocco: 'MA', algeria: 'DZ', ethiopia: 'ET',
  'ivory coast': 'CI', ghana: 'GH', senegal: 'SN', israel: 'IL',
  'saudi arabia': 'SA', 'united arab emirates': 'AE', iraq: 'IQ', iran: 'IR',
  kazakhstan: 'KZ', uzbekistan: 'UZ', mongolia: 'MN', nepal: 'NP', myanmar: 'MM',
  burma: 'MM', cambodia: 'KH', laos: 'LA', taiwan: 'TW', 'hong kong': 'HK',
  iceland: 'IS', estonia: 'EE', latvia: 'LV', lithuania: 'LT', croatia: 'HR',
  serbia: 'RS', bulgaria: 'BG', slovenia: 'SI', 'bosnia': 'BA', albania: 'AL',
  'north macedonia': 'MK', macedonia: 'MK',
};

/** A country code from a JSON country field (ISO code or full name), or from
 *  a full country name mentioned in free text. */
function countryCode(value: string): string | undefined {
  const v = value.trim();
  if (/^[A-Za-z]{2}$/.test(v)) {
    const code = v.toUpperCase();
    return COUNTRY_CENTROIDS[code] !== undefined ? code : undefined;
  }
  return COUNTRY_NAMES[v.toLowerCase()];
}

/** A country mentioned by full name in the model's text. */
function mentionedCountry(text: string): string | undefined {
  const lower = ` ${text.toLowerCase()} `;
  let best: string | undefined;
  let bestLen = 0;
  for (const [name, code] of Object.entries(COUNTRY_NAMES)) {
    if (lower.includes(` ${name} `) && name.length > bestLen) {
      best = code;
      bestLen = name.length;
    }
  }
  return best;
}

/** One model call through the Anthropic gateway; null when refused (final). */
async function askAnthropic(jpeg: Buffer, prompt: string): Promise<string | null> {
  const body = JSON.stringify({
    model: ANTHROPIC_MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: prompt },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${anthropicUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': anthropicKey, 'anthropic-version': '2023-06-01' },
      body,
    });
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
    }
    const text = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`agent: anthropic call refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
    return null;
  }
}

/** One model call through the OpenAI gateway; null when refused (final). */
async function askOpenAI(jpeg: Buffer, prompt: string): Promise<string | null> {
  const body = JSON.stringify({
    model: OPENAI_MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${jpeg.toString('base64')}` } },
          { type: 'text', text: prompt },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${openaiUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${openaiKey}` },
      body,
    });
    if (res.ok) {
      const msg = (await res.json()) as { choices?: { message?: { content?: string } }[] };
      return msg.choices?.[0]?.message?.content ?? '';
    }
    const text = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`agent: openai call refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
    return null;
  }
}

/** Ask the vision model where a photo was taken: Anthropic first, OpenAI fallback. */
async function ask(jpeg: Buffer, prompt: string): Promise<string | null> {
  if (anthropicUrl && anthropicKey) {
    const text = await askAnthropic(jpeg, prompt);
    if (text !== null) return text;
  }
  if (openaiUrl && openaiKey) {
    return askOpenAI(jpeg, prompt);
  }
  return null;
}

/** Post one photo's answer to the env; returns true when accepted. */
async function postAnswer(g: Guess): Promise<boolean> {
  const res = await fetch(`${envUrl}/answers`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ answers: [g] }),
  });
  if (!res.ok) {
    process.stderr.write(`agent: post ${g.id} refused (HTTP ${res.status}): ${(await res.text()).slice(0, 200)}\n`);
    return false;
  }
  return true;
}

/** Guess one photo: model call, second chance when needed, country-centroid
 *  fallback when the model names a country but gives no coordinates. */
async function guessPhoto(p: Photo): Promise<Guess | null> {
  const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
  let text = await ask(jpeg, PROMPT);
  let g = text === null ? null : parseGuess(text);
  if ((g === null || !Number.isFinite(g.lat)) && text !== null) {
    // No coordinates in the first answer: one simpler retry.
    const retry = await ask(jpeg, SECOND_CHANCE_PROMPT);
    if (retry !== null) {
      const rg = parseGuess(retry);
      if (rg !== null && Number.isFinite(rg.lat)) g = rg;
      else if (rg !== null && rg.country !== undefined) g = rg;
      else text = retry;
    }
  }
  if (g !== null && Number.isFinite(g.lat)) return { id: p.id, lat: g.lat, lon: g.lon };
  // Last resort: the model named a country — post its centroid.
  const country = g?.country ?? (text !== null ? mentionedCountry(text) : undefined);
  if (country !== undefined) {
    const c = COUNTRY_CENTROIDS[country];
    if (c !== undefined) {
      process.stderr.write(`agent: ${p.id}: no coordinates, falling back to ${country} centroid\n`);
      return { id: p.id, lat: c[0], lon: c[1] };
    }
  }
  if (text !== null) process.stderr.write(`agent: ${p.id}: no usable answer: ${text.slice(0, 200)}\n`);
  return null;
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  let answered = 0;
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      try {
        const g = await guessPhoto(p);
        if (g !== null && (await postAnswer(g))) answered += 1;
      } catch (err) {
        process.stderr.write(`agent: ${p.id} failed: ${(err as Error).message}\n`);
      }
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  process.stdout.write(`agent: ${answered} of ${photos.length} photos answered\n`);
  return answered === 0 ? 1 : 0;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`agent: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
