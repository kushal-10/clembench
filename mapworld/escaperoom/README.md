# ESCAPE ROOM

## Introduction

#### Game Details
Escape Room is a multimodal collaborative game between two agents based on the Mapworld Environment. Here, one agent - PLAYERA is stuck inside the mapworld, who can explore the world but has limited observability (i.e. can only observe the current room and possible exits at each time step) (Can add large enough memory part #TODO). There's another agent, PLAYERB outside the mapworld, with no observability and cannot explore the world. PLAYERB has been given only an Image of an Escape Room. PLAYERA needs to reach this escape room to win the Game. PLAYERA and PLAYERB can only communicate with each other via a text channel

As shown in the below figure, Game starts with B sending the description of the Escape Room. A checks if its the same room, If he's sure, he can say the tag "ESCAPING" and the game ends there. If he's not sure, he can ask for more deatils starting with tag "DESCRIPTION:". Then, B needs to provide it, by using the same tag "DESCRIPTION:" in a greater depth, or include minor details (Useful only in Ambiguous cases). If A is definitely sure that this is not the room, he can discuss it with B and ask for move suggestions using "SUGGESTION:" as a tag, a B can suggest a move to A

![Captiones](flow.png)
