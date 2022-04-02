import json
import pandas as pd
import numpy as np
import re

# Return a list of words (because some entries have more than one word)
# Lower all words, remove _ and ' ", remaning only words
def clean (words):
    temp = words.split(',')
    list_words = list()
    for word in temp:
        # Remove any wrong "word" created 
        if word in ['_', '"', "'", ' ', "signal_1"]:
            continue

        # Remove any comment created in () or e.g ()
        after = re.sub(r'e*\.*g*\.*\s*\(.*\)', ' ', word)

        # Turn all words lower, replace _ used as spaces, turn " and ' in spaces, then removing all spaces before and after the word
        list_words.append(after.lower().replace('_', ' ').replace('"', ' ').replace("'", " ").strip())
    return list_words


if __name__ == "__main__":
    signs =  dict()
    arch = []
    # Read all tsv files and append them as Pandas Dataframes to a list
    for i in range (1, 10):
        arch.append(pd.read_csv(f"{i}.tsv", sep="\t"))

    # '.', ',' and ';' were used by revisor 2 to agree with revisor 1 and/or translation
    agreed = [".", ",", ";", ".."]
    sign = 0
    for file in arch:
        # keep track of which sign were are treating
        sign += 1
        temp = []

        # Add all valid options in a list, then in a dictionary
        # Maybe create a new system to filter better
        for i in file.index:
            translation = file['Traduzido'][i]
            rev1 = file['Sugestões Jair'][i]
            rev2 = file['Sugestões Gabriel'][i]
            if not pd.isnull(translation):
                if pd.isnull(rev1) and not pd.isnull(rev2): 
                    if rev2 in agreed:
                        for word in clean(translation):
                            temp.append(word)
                    else:
                        for word in clean(translation):
                            temp.append(word)
                        for word in clean(rev2):
                            temp.append(word)
                elif not pd.isnull(rev1) and not pd.isnull(rev2):
                    if rev2 in agreed:
                        for word in clean (translation):
                            temp.append(word)
                        for word in clean (rev1):
                            temp.append(word)
                    else:
                        for word in clean (translation):
                            temp.append(word)
                        for word in clean (rev1):
                            temp.append(word)
                        for word in clean (rev2):
                            temp.append(word)
                elif not pd.isnull(rev1) and pd.isnull(rev2):
                    for word in clean(translation):
                        temp.append(word)
                    for word in clean(rev1):
                        temp.append(word)
                else:
                    continue
            else:
                if pd.isnull(rev1) and pd.isnull(rev2):
                    continue
                elif pd.isnull(rev1) and not pd.isnull(rev2):
                    if rev2 in agreed:
                        continue
                    for word in clean (rev2):
                        temp.append(word)
                elif not pd.isnull(rev1) and pd.isnull(rev2):
                    for word in clean(rev1):
                        temp.append(word)
                else:
                    for word in clean(rev1):
                        temp.append(word)
                    if rev2 in agreed:
                        continue
                    for word in clean(rev2):
                        temp.append(word)

            # Use a set instead of a list to remove duplications, then re-making a list to create a json
            signs[sign] = list(set(temp))
    with open('signs.json', 'w', encoding='utf-8') as f:
        json.dump(signs, f, ensure_ascii=False, indent=4)
