"""
Example Scenario

What does the game actually do

Player A outside the world

######
- Init Prompt on the lines of = 
You are given an image/a description of a room in the mapworld. 
Another agent is stuck in the mapworld and needs to reach the room that you have a description/an image of to escape the mapworld.
Your task is to help the Agent stuck in the mapworld and help the agent to reach the escaperoom
You can only communicate via a text-medium, so you need to describe the image as best as you can to the Agent

You have no additional information about the environment, what are the neighboring rooms, how many rooms does the environment has etc..
You are only given an Image.

Always start your response with the tag <DESCRIPTION>: your description

You will get the description back from the agent. Then your task is to verify if its the same room or not, then you should only reply with
"I think you are in the right room" OR "I dont think you are in the right room"
Begin your response with the tag <VERIFICATION>: response.
######
Maybe add a <REMINDER> Tag if using "memory" to remind the agent that it has already visited this room before...


Agent Prompt
######
You are stuck in a mapworld environment. Your task is to reach an escape room, in this world but you dont know which room it is.
There is a Guide agent outside this world who knows how the escape room looks like. You can communicate with the Guide only via a text channel
Start by describing the current room you are in to the Guide and wait for the response. 
If the guide responds by saying that you are in the correct room, then Send the message starting with the tag <ESCAPING>: "Trying to escape form {this_room}"
If the guide responds otherwise, make a move to another room from the given options and describe that room by saying <DESCRIPTION>: description
######

Maybe have multiple turns on the same room with different tags...
"""


"""
Defining an instance

1) Player A/ Player B
Player A - Inside the world, Player B - Outside the world...

Both are given an initial image, image for player B is fixed, image for Player A changes at every step


"""