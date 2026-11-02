import traceback
try:
    exec(open('_gen_doc.py', encoding='utf-8').read())
except Exception as e:
    traceback.print_exc()
