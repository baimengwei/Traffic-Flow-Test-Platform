import json
import engine
import os


print(os.getcwd())
print(os.listdir(os.getcwd()))

file = open('config.json')
content = json.load(file)
content = json.dumps(content)
file.close()


eng = engine.Engine(1,1,True,True)
eng.load_roadnet('roadnet.json')
eng.load_flow('flow.json')
for i in range(100):
    eng.next_step()
