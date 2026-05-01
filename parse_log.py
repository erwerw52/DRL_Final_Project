import json
import sys

try:
    with open('conversation.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out = []
    for line in lines:
        if not line.strip(): continue
        try:
            data = json.loads(line)
        except:
            continue
            
        if data.get('source') == 'USER_EXPLICIT' and 'content' in data:
            content = data['content']
            if '<USER_REQUEST>' in content:
                content = content.split('<USER_REQUEST>')[1].split('</USER_REQUEST>')[0].strip()
            out.append(f'## User\n\n{content}\n\n---\n')
        elif data.get('source') == 'MODEL' and 'content' in data:
            content = data['content']
            out.append(f'## Antigravity (AI)\n\n{content}\n\n---\n')

    with open('conversation.md', 'w', encoding='utf-8') as f:
        f.write('# Conversation History\n\n')
        f.write('\n'.join(out))
    print('Converted to readable markdown successfully.')
except Exception as e:
    print('Error:', e)
