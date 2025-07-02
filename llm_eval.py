from llm_server import get_response

# calling the server for results in different domains 

# All three models did well on this task
#print(get_response("The radius of a cirlce is 1. Calculate the area of the circle and write it in C++."))

# Only the backup model was able to answer correctly
#print(get_response("Can you summarize the biography of Freddie Mercury in five sentences?"))

print(get_response("""Which of the following technological advancements is **not** typically associated with the Second Industrial Revolution?
A: The light bulb
B: The telephone
C: The assembly line
D: The steam engine
E: The automobile
F: The radio
G: The airplane
H: The electric motor
I: The phonograph
J: The refrigerator"""))