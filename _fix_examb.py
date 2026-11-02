with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Replace lines 4036-4058 (0-indexed: 4035-4057)
new_block = [
    "    questions = None\n",
    "\n",
    "    # \u2500\u2500 Generation via google.genai (gemini-2.0-flash) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "    try:\n",
    "        from google import genai as google_genai\n",
    "        gclient  = google_genai.Client(api_key=GEMINI_API_KEY)\n",
    "        response = gclient.models.generate_content(\n",
    "            model='gemini-2.0-flash',\n",
    "            contents=prompt,\n",
    "        )\n",
    "        raw = response.text.strip()\n",
    "        raw = re.sub(r'^```(?:json)?\\s*', '', raw, flags=re.MULTILINE)\n",
    "        raw = re.sub(r'```\\s*$', '', raw, flags=re.MULTILINE)\n",
    "        arr_match = re.search(r'\\[.*\\]', raw, re.DOTALL)\n",
    "        if arr_match:\n",
    "            questions = json.loads(arr_match.group())\n",
    "    except Exception as e:\n",
    '        print(f"[TinyLlama] Inference error: {str(e)[:200]}")\n',
]

# lines are 1-indexed; slice is 0-indexed
# replace lines 4036 to 4058 inclusive (indices 4035 to 4057)
lines[4035:4058] = new_block

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("DONE")
