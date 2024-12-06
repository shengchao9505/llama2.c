



fout = open("data.bin", "wb")


for i in ...:
  # patient-i
  a = [snp_token_ids...]
  a = np.array([...], dtype=np.uint32)
  fout.write(a.tobytes())

fout.close()

