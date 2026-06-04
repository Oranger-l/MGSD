# Case 06: minibehaviour level5

- case_id: `minibehaviour_test_level5_0028`
- task: `minibehaviour`
- level: `5`
- input_modality: `text`
- prompt_file: `prompts/text_student_MiniBehaviour.txt`

## Final User Prompt

```text
You are a MiniBehaviour solver.

Task:
Analyze the text description and produce a valid action plan. The agent must go to the printer, pick it up, go to the table, and drop the printer.

Text input:
You will receive the grid size, agent position, printer position, table cells, printer-adjacent cells, table-adjacent cells and a text grid as text.

Rules:
1. The grid uses 0-based (row, column) coordinates.
2. Valid movement actions are L (Left), D (Down), R (Right), and U (Up).
3. The agent cannot enter any table cell. The printer cell is blocked before PICK, but after PICK the printer is removed and that cell becomes traversable.
4. PICK is legal only when the agent is adjacent to the printer.
5. DROP is legal only after PICK, when the agent is adjacent to the table.
6. The plan must perform PICK before DROP and finish immediately after a legal DROP.

Output requirements:
1. First, briefly state the grid size and the positions of the Agent, Printer, and Table.
2. Next, give exactly one short sentence that states the planned route at a high level and mentally verifies that PICK and DROP are legal.
3. Do not narrate the solution step by step. Do not list repeated moves, repeated coordinates, or intermediate states outside <answer>.
4. End with exactly one <answer>...</answer> block containing only comma-separated actions, for example: <answer>L,L,PICK,D,D,R,DROP</answer>

Example Format:
The grid size is 5x5. The agent starts at (4,0), the printer is at (3,1), and the table cells are: (1,2); (1,3); (1,4); (2,2); (2,3); (2,4). To complete the task, we move up from (4,0) to (3,0), which is adjacent to the printer at (3,1), then PICK legally; after PICK, we move right into the now-empty printer cell (3,1), then right to (3,2), which is adjacent to table cell (2,2), and DROP legally. So the final answer is <answer>U,PICK,R,R,DROP</answer>

Please generate the action plan for the following text-described MiniBehaviour grid:

Text state:
Task: MiniBehaviour
Grid size: 5x5
Agent position: (2,1)
Printer position: (1,2)
Table cells: (2,2); (2,3); (2,4); (3,2); (3,3); (3,4)
Grid legend: A=agent, P=printer, T=table, .=free floor
Grid:
.....
..P..
.ATTT
..TTT
.....
Cells adjacent to the printer where PICK is legal: (0,2); (1,1); (1,3)
Cells adjacent to the table where DROP is legal after PICK: (1,2); (1,3); (1,4); (2,1); (3,1); (4,2); (4,3); (4,4)
The agent may move through any in-bounds non-table cell.
The printer cell is blocked before PICK; after PICK, the printer is removed and that cell becomes traversable.

```
