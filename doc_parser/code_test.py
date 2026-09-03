strs=["hel", "hellodddddddd", "ddddIndia"]
st=""
for i in range(len(strs)):
    if len(st)<30 and len((st+strs[i]))<30:
        st+=f"\n {strs[i]}"
    print(st)
    st=""


print(len(strs))

