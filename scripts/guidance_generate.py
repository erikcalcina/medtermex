import warnings
import json
from scripts.prompts import prompts

warnings.filterwarnings("ignore")

from guidance import models, gen, select


class generate:

    def gene(model, prompt, max_tokens=250):
        lm = model + prompt
        lm += "\n" + gen("pred", max_tokens=max_tokens, stop="}") + "}"
        return lm["pred"]

        
    def json_format(pred):
        sep = "}"
        pred = pred.split(sep, 1)[0] + sep
        try:
            json_pred = json.loads(pred)
        except:
            print("Something went wrong with json.loads(pred)")
            #print(pred)
            return {}
        else:
            json_pred = json.loads(pred)
            return json_pred

    def extract(model, entities, text, max_length=2000):
        output = {}
        for entity in entities:
            try:
                prompt = getattr(prompts, entity.lower())(text)
                output.update(generate.json_format(generate.gene(model, prompt, max_length)))
            except Exception as e:
                print(e)
                continue

        return output