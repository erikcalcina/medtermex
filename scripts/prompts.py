
global start 
#start = "### Your goal is to extract structured information from the user's input that matches json format. When extracting information please make sure it matches the type information exactly. Extract the following entity: "
start = "### Your goal is to extract structured information from the user's input that matches json format. When extracting information please make sure it matches the type information exactly. Extract the following entities: "

class prompts:

    def age(input):
        #example = {"age": ["32-year-old"]}
        age = start + f"Age. Do not add any extra entities.\n\n### Input:\n{input}\n\n### Output:"
        #age = start + f"Age. Do not add any extra entities. Example: {{\"Age\": [\"32-year-old\"]}}\n\n### Input:\n{input}\n\n### Output:"
        #age = start + f"Age. Do not add any extra entities. Example: {example}\n\n### Input:\n{input}\n\n### Output:"
        #print(age)
        return(age)
        
    def sex(input):
        sex = start + f"Sex. Do not add any extra entities.\n\n### Input:\n{input}\n\n### Output:"
        #print(sex)
        return(sex)
        
    def biological_structure(input):
        bio = start + f"Biological_structure. Do not add any extra entities.\n\n### Input:\n{input}\n\n### Output:"
        #bio = start + f"Biological_structure. Do not add any extra entities. Example: {{\"Biological_structure\": [\"intraperitoneal fluid\", \"liver\", \"pelvis\", \"rectal\"}}\n\n### Input:\n{input}\n\n### Output:"
        #print(bio)
        return(bio)
        
    def sign_symptom(input):
        sign = start + f"Sign_symptom. Do not add any extra entities.\n\n### Input:\n{input}\n\n### Output:"
        #print(sign)
        return(sign)
    
    def diagnostic_procedure(input):
        diagn = start + f"Diagnostic_procedure. Do not add any extra entities.\n\n### Input:\n{input}\n\n### Output:"
        #print(diagn)
        return(diagn)

    def lab_value(input):
        lab = start + f"Lab_value. Do not add any extra entities.\n\n### Input:\n{input}\n\n### Output:"
        #print(lab)
        return(lab)

#hah = prompts.age("to je moj prompt")
#print(hah)
    