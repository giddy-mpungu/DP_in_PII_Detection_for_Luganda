// Rule-based USERID and DATE recognisers for natural (typed) text, adapted from experiments/hybrid_eval.py,
// whose word-piece version was the one evaluated in the dissertation (Section 4.8). Differences: they work on
// raw text, accept spaced/hyphenated phone numbers and emails, and mark whole words.

const WEEKDAY = /^(o?lw(a|o)?(kubiri|kusatu|kuna|kutaano|mukaaga)|ssande|sande|mmande|bbalaza|monday|tuesday|wednesday|thursday|friday|saturday|sunday)$/;
const ORDINAL = /^(okubiri|okusatu|okuna|okutaano|omukaaga|omusanvu|omunaana|omwenda|ekkumi|olubereberye)$/;
const MONTH = /^(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|januari|janwali|febwali|maaki|apuli|maayi|juuni|j?julaayi|agusito|sebutemba|okitobba|novemba|desemba|museenene)$/;
const YEAR = /^(19[5-9]\d|20[0-3]\d)$/;
const DAY = /^(0?[1-9]|[12]\d|3[01])$/;
const CURRENCY = new Set(["ssiringi", "shs", "ugx", "ssente", "obukadde", "zokka", "doola"]);
const MODIFIER = new Set(["eno", "ejja", "ewedde", "eyise"]);
const WEEK = new Set(["wiiki", "ssabbiiti", "sabbiiti", "wiikendi"]);
const SINGLE = { th: "T", st: "T", nd: "T", rd: "T", "/": "P", ".": "P", "-": "P", ",": "C", mwezi: "K", omwezi: "K",
  gwa: "G", ogwa: "G", gw: "Q", lw: "Q", "'": "A", "’": "A", lwa: "L", nga: "N", mu: "U", oluwedde: "R", "(": "(" };

// Same grammar as hybrid_eval.DATE_PATTERN (one class letter per word)
const WD = "(?:W|QAX)", MON = "(?:KG?M|K?QAX|M(?:\\(M)?)";
const DM = `(?:DT?(?:P(?:DT?|M))?C?U?(?:KG)?${MON})`, NUM = "(?:DP(?:D|M)P(?:Y|D)|DPM)";
const CORE = `(?:${NUM}|${DM}|${MON}|${WD}|Y(?:PY)?)`;
const TAIL = `(?:LVE|R|C?Y|N(?:${NUM}|${DM}|MD|DT?)|U${MON})*`;
const DATE_PATTERN = new RegExp(CORE + TAIL, "g");

function wordClass(words, i) {
  const w = words[i].text, prev = i ? words[i - 1].text : "", next = i + 1 < words.length ? words[i + 1].text : "";
  if (WEEKDAY.test(w)) return "W";
  if (MONTH.test(w)) return "M";
  if (YEAR.test(w) && !CURRENCY.has(prev) && !CURRENCY.has(next)) return "Y";
  if (DAY.test(w)) return "D";
  if (ORDINAL.test(w) && i >= 2 && SINGLE[words[i - 1].text] === "A" && ["gw", "lw"].includes(words[i - 2].text)) return "X";
  if (w in SINGLE) return SINGLE[w];
  return MODIFIER.has(w) ? "E" : WEEK.has(w) ? "V" : "_";
}

const USERID_PATTERNS = [
  /[\p{L}\p{N}._%+-]+@[\p{L}\p{N}-]+(?:\.[\p{L}\p{N}-]+)+/gu,                 // email
  /(?<![\p{L}\p{N}])(?:\+?256[\s-]?|0)7\d{2}[\s-]?\d{3}[\s-]?\d{3}(?![\p{N}])/gu, // Ugandan mobile number
  /(?<![\p{L}\p{N}])[a-z]{2}\d{7,12}[a-z0-9]{2,4}(?![\p{L}\p{N}])/giu,          // national-ID-like
  /(?<![\p{L}\p{N}])u[a-z]{2}\s?\d{3}[a-z]?(?![\p{L}\p{N}])/giu,                 // number plate
  /(?<![\p{L}\p{N}])\+?\d{6,13}(?![\p{N}])/gu,                                   // other long numbers
];

// words: [{text (lower-cased), start, end}] as produced by app.js; returns [{type, start, end}] in characters
export function ruleSpans(text, words) {
  const out = [];
  for (const re of USERID_PATTERNS) {
    for (const m of text.matchAll(re)) {
      const s = m.index, e = m.index + m[0].length;
      if (!out.some((o) => s < o.end && e > o.start)) out.push({ type: "USERID", start: s, end: e, source: "rule" });
    }
  }
  const cls = words.map((_, i) => wordClass(words, i)).join("");
  for (const m of cls.matchAll(DATE_PATTERN)) {
    if (!/[WMYX]/.test(m[0])) continue;
    const s = words[m.index].start, e = words[m.index + m[0].length - 1].end;
    if (!out.some((o) => s < o.end && e > o.start)) out.push({ type: "DATE", start: s, end: e, source: "rule" });
  }
  return out;
}
