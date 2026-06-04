# Case 03: maze level3

- case_id: `maze_test_level3_0000`
- task: `maze`
- level: `3`
- input_modality: `text`
- prompt_file: `prompts/text_student_Maze.txt`

## Final User Prompt

```text
You are a Maze solver.

Task:
Analyze the text description and produce a valid action plan from the player to the goal without crossing any wall.

Text input:
You will receive the map size, start and target coordinates, an open-direction table for every cell.

Rules:
1. The grid uses 0-based (row, column) coordinates.
2. Valid actions are L (Left), D (Down), R (Right), and U (Up).
3. A move is legal only when the current cell lists that direction as open.
4. The route should finish at the target without crossing any wall.

Output requirements:
1. First, briefly state the map size, the positions of the Player and Goal, and an open-direction table for every cell.
2. Next, give exactly one short sentence that states the planned route at a high level and mentally verifies that it reaches the Goal without crossing walls.
3. Do not narrate the solution step by step. Do not list repeated moves, repeated coordinates, or intermediate states outside <answer>.
4. End with exactly one <answer>...</answer> block containing only the complete action plan using L, D, R, U, for example: <answer>DDRUR</answer>

Example Format:
The map size is 2x2. The player starts at (0,0), and the goal is at (1,1). Open directions: (0,0): right, down; (0,1): left, down; (1,0): up; (1,1): up. To reach the goal, we move right from (0,0) to (0,1), then down to (1,1), and both moves follow open directions without crossing any wall. So the final answer is <answer>RD</answer>

Please generate the action plan for the following text-described maze:

Text state:
Task: Maze
Map size: 3x3
Start position: (0,0)
Target position: (0,2)
Open directions:
(0,0): down
(0,1): right, down
(0,2): down, left
(1,0): up, right, down
(1,1): up, left
(1,2): up
(2,0): up, right
(2,1): right, left
(2,2): left

```
