# Qualitative samples (dev set)

## Example 1 - CORRECT  (dev line 4919)
- **Question:** WHAT IS THE SCORE WITH A RECORD OF 1-0?
- **Gold SQL:** `SELECT Score FROM table WHERE Record = '1-0'`
- **Our SQL:** `SELECT Score FROM table WHERE Record = '1-0'`

## Example 2 - CORRECT  (dev line 5341)
- **Question:** What was the method of resolution for the fight against dale hartt?
- **Gold SQL:** `SELECT Method FROM table WHERE Opponent = 'dale hartt'`
- **Our SQL:** `SELECT Method FROM table WHERE Opponent = 'dale hartt'`

## Example 3 - CORRECT  (dev line 525)
- **Question:** Who is the courtside reporter for the year 2009-10?
- **Gold SQL:** `SELECT Courtside reporter FROM table WHERE Year = '2009-10'`
- **Our SQL:** `SELECT Courtside reporter FROM table WHERE Year = '2009-10'`

## Example 4 - CORRECT  (dev line 3380)
- **Question:** Opponent of @ edmonton oilers, and a Game larger than 1, and a Series of oilers lead 3–2 had what score?
- **Gold SQL:** `SELECT Score FROM table WHERE Opponent = '@ edmonton oilers' AND Game > 1 AND Series = 'oilers lead 3–2'`
- **Our SQL:** `SELECT Score FROM table WHERE Game > 1 AND Series = 'oilers lead 3–2' AND Opponent = '@ edmonton oilers'`

## Example 5 - CORRECT  (dev line 6366)
- **Question:** What's the venue for the home team that scored 9.14 (68)?
- **Gold SQL:** `SELECT Venue FROM table WHERE Home team score = '9.14 (68)'`
- **Our SQL:** `SELECT Venue FROM table WHERE Home team score = '9.14 (68)'`

## Example 6 - WRONG  (dev line 5701)
- **Question:** What year is the happy planet index?
- **Gold SQL:** `SELECT SUM(Year) FROM table WHERE Index = 'happy planet index'`
- **Our SQL:** `SELECT Year FROM table WHERE Index = 'happy'`
- **Failure:** wrong aggregation, wrong value

## Example 7 - WRONG  (dev line 4567)
- **Question:** What is the tries against when the points are 475?
- **Gold SQL:** `SELECT Tries against FROM table WHERE Points for = '475'`
- **Our SQL:** `SELECT Tries for FROM table WHERE Points for = '475'`
- **Failure:** wrong select column

## Example 8 - WRONG  (dev line 3331)
- **Question:** When the other is n/a and the psc-pc is 30% what is the date?
- **Gold SQL:** `SELECT Date FROM table WHERE Others = 'n/a' AND PSD - PC = '30%'`
- **Our SQL:** `SELECT Date FROM table WHERE Undecided = 'n/a' AND PSD - PC = '30%'`
- **Failure:** wrong condition column/operator

## Example 9 - WRONG  (dev line 5550)
- **Question:** What is the date of the game when attendance is more than 20,682?
- **Gold SQL:** `SELECT Date FROM table WHERE Attendance > '20,682'`
- **Our SQL:** `SELECT Date FROM table WHERE Attendance > '20,602'`
- **Failure:** wrong value

## Example 10 - WRONG  (dev line 4031)
- **Question:** What is the Average annual output for Culligran power station with an Installed capacity less than 19?
- **Gold SQL:** `SELECT AVG(Average annual output (million KWh)) FROM table WHERE Name = 'culligran' AND Installed capacity (megawatts) < 19`
- **Our SQL:** `SELECT AVG(Average annual output (million KWh)) FROM table WHERE Name = 'culligran power station' AND Installed capacity (megawatts) < 19`
- **Failure:** wrong value
