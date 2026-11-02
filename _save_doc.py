"""Append the final sections and save the docx."""
import sys, os

# Read the existing gen script and exec it up to the page break
content = open('_gen_doc.py', encoding='utf-8').read()

# Execute everything already in _gen_doc.py
exec(content, {'__name__': '__main__'})
