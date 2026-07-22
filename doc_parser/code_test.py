all_chapters=["sdfdsfdfd","dfsdff","dfddfdf"]
all_chap=[]
for i in all_chapters:
        if all_chapters.index(i)==0:
            all_chap.append(f"# **Nationalism in Bengaluru** "+"\n" + i) 
        else:
            all_chap.append(i)
print(all_chap[0])  