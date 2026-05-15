import pathlib
p=pathlib.Path('static/css/style.css')
text=p.read_text(encoding='utf-8')
opens=text.count('{')
closes=text.count('}')
print('opens',opens,'closes',closes,'diff',opens-closes)
stack=[]
lines=text.splitlines()
for i,line in enumerate(lines,1):
    for ch in line:
        if ch=='{': stack.append((i,ch))
        elif ch=='}':
            if stack: stack.pop()
            else: print('extra close at',i)
if stack: print('unmatched open at',stack[-1])
