import os
TESTDATA_PATH = './data/bangla/Testing'
for data in os.walk(TESTDATA_PATH):
  test_data=data[2]

  print(test_data)