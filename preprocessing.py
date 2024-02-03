
import numpy as np
import re
class Preprocessing:
	
	@staticmethod
	def read_dataset(file):
		
		letters = ['ሀ','ሁ','ሂ','ሃ', 'ሄ', 'ህ', 'ሆ','ለ', 'ሉ', 'ሊ', 'ላ', 'ሌ', 'ል',
             'ሐ', 'ሑ' , 'ሒ' , 'ሓ', 'ሔ', 'ሕ', 'ሖ','መ', 'ሙ', 'ሚ', 'ማ', 'ሜ', 'ም','ሠ', 'ሡ','ሢ','ሣ' , 'ሤ','ሥ','ሦ','ረ','ሩ',
             'ሪ','ራ', 'ሬ', 'ር','ሰ', 'ሱ', 'ሲ', 'ሳ', 'ሴ','ስ', 'ሶ','ሸ', 'ሹ','ሺ', 'ሻ' ,'ሼ','ሽ','ሾ','ቀ','ቁ','ቂ','ቃ','ቄ','ቅ','ቆ','በ','ቡ','ቢ','ባ','ቤ', 'ብ','ቦ',
'ተ','ቱ','ቲ','ታ','ቴ','ት', 'ቶ',
'ቸ', 'ቹ', 'ቺ', 'ቻ', 'ቼ', 'ች', 'ቾ',
'ኀ', 'ኁ', 'ኂ', 'ኃ', 'ኄ', 'ኅ', 'ኆ',
'ነ', 'ኑ', 'ኒ', 'ና', 'ኔ', 'ን',
'ኘ' ,'ኙ', 'ኚ', 'ኛ', 'ኜ', 'ኝ',
'አ', 'ኡ', 'ኢ', 'ኣ', 'ኤ','እ', 'ኦ',
'ከ', 'ኩ', 'ኪ', 'ካ', 'ኬ', 'ክ', 'ኮ',
'ኸ', 'ኹ', 'ኺ', 'ኻ', 'ኼ', 'ኽ', 'ኾ',
'ወ', 'ዉ', 'ዊ', 'ዋ', 'ዌ', 'ው','ዎ',
'ዐ', 'ዑ', 'ዒ', 'ዓ', 'ዔ','ዕ','ዖ',
'ዘ', 'ዙ', 'ዚ','ዛ', 'ዜ','ዝ', 'ዞ',
'ዠ','ዡ', 'ዢ', 'ዣ', 'ዤ', 'ዥ', 'ዦ',
'የ', 'ዩ', 'ዪ', 'ያ', 'ዬ', 'ይ', 'ዮ',
'ደ' ,'ዱ', 'ዲ', 'ዳ', 'ዴ', 'ድ', 'ዶ',
'ጀ', 'ጁ', 'ጂ', 'ጃ', 'ጄ', 'ጅ', 'ጆ',
'ገ', 'ጉ', 'ጊ' , 'ጋ', 'ጌ', 'ግ', 'ጎ',
'ጠ', 'ጡ', 'ጢ' , 'ጣ', 'ጤ', 'ጥ','ጦ',
'ጨ', 'ጩ', 'ጪ', 'ጫ', 'ጬ', 'ጭ','ጮ',
'ጰ', 'ጱ', 'ጲ', 'ጳ', 'ጴ' , 'ጵ', 'ጶ',
'ጸ', 'ጹ','ጺ', 'ጻ', 'ጼ', 'ጽ', 'ጾ',
'ፀ', 'ፁ','ፂ', 'ፃ', 'ፄ', 'ፅ', 'ፆ',
'ፈ', 'ፉ', 'ፊ', 'ፋ', 'ፌ', 'ፍ', 'ፎ',
'ፐ', 'ፑ', 'ፒ', 'ፓ', 'ፔ', 'ፕ', 'ፖ', ' ']
		
		# Open raw file
		with open(file, 'r',encoding='utf-8') as f:
			raw_text = f.readlines()
			
		# Transform each line into lower
		raw_text = [line.lower() for line in raw_text]
		
		# Create a string which contains the entire text
		text_string = ''
		for line in raw_text:
			text_string += line.strip()
		
		# Create an array by char
		text = list()
		for char in text_string:
			text.append(char)
	
		# Remove all symbosl and just keep letters
		text = [char for char in text if char in letters]
	
		return text
		
	@staticmethod
	def create_dictionary(text):
		
		char_to_idx = dict()
		idx_to_char = dict()
		
		idx = 0
		for char in text:
			if char not in char_to_idx.keys():
				char_to_idx[char] = idx
				idx_to_char[idx] = char
				idx += 1
				
		print("Vocab: ", len(char_to_idx))
		
		return char_to_idx, idx_to_char
		
	@staticmethod
	def build_sequences_target(text, char_to_idx, window):
		
		x = list()
		y = list()
	
		for i in range(len(text)):
			try:
				# Get window of chars from text
				# Then, transform it into its idx representation
				sequence = text[i:i+window]
				sequence = [char_to_idx[char] for char in sequence]
				
				# Get char target
				# Then, transfrom it into its idx representation
				target = text[i+window]
				target = char_to_idx[target]
				
				# Save sequences and targets
				x.append(sequence)
				y.append(target)
			except:
				pass
		
		x = np.array(x)
		y = np.array(y)
		
		return x, y
		
