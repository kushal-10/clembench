# EscapeGame

---

`EscapeGame` is a multi-player game with two players - `Guide` and `Seeker`, with the Seeker inside the MapWorld environment (Refer Engine) and the Guide outside the environment. The task for them is to collaborate with each other via a text channel and help the Seeker reach an `Escape Room`. 

The Guide is provided with the image of the escape room, and the game starts with the Guide generating a description of the escape room. This description is passed to the Seeker. Along with this, the Seeker is provided with an image of the current room it is in. With the description of the escape room and the image of the current room, the Seeker can respond with one of the following:

- `MOVE: <direction>` - If the Seeker believes that the description of the escape room does not match with the image it is given
- `QUESTION: <question>` - If the Seeker believes that the description somewhat matches with its current image and it needs more information to verify.
- `ESCAPE` - If the Seeker believes that the description matches exactly with the image of its current room

The Seeker acts as a proactive agent -  it can interact with the environment, ask questions or decide to escape. The Guide, on the other hand, acts as a reactive agent here, only responding with `ANSWER: <answer>` whenever the Seeker asks a `QUESTION`. The Guide cannot interact with the environment or initiate a dialog with the Seeker.

# Prompts
---

(ADD - file locations of the prompts)

# Instance Generation

TBC....


