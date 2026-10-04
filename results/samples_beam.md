# Qualitative samples (dev set)

## Example 1 - CORRECT  (dev line 4919)
- **Question:** WHAT IS THE SCORE WITH A RECORD OF 1-0?
- **Gold SQL:** `SELECT Score FROM table WHERE Record = '1-0'`
- **Our SQL:** `SELECT Score FROM table WHERE Record = '1-0'`

## Example 2 - CORRECT  (dev line 5333)
- **Question:** What is the record for the St. Louis team?
- **Gold SQL:** `SELECT Record FROM table WHERE Team = 'st. louis'`
- **Our SQL:** `SELECT Record FROM table WHERE Team = 'st. louis'`

## Example 3 - CORRECT  (dev line 524)
- **Question:** What are all sail numbers for the yacht Yendys?
- **Gold SQL:** `SELECT Sail number FROM table WHERE Yacht = 'Yendys'`
- **Our SQL:** `SELECT Sail number FROM table WHERE Yacht = 'yendys'`

## Example 4 - CORRECT  (dev line 3376)
- **Question:** Which visitor has a Los Angeles home?
- **Gold SQL:** `SELECT Visitor FROM table WHERE Home = 'los angeles'`
- **Our SQL:** `SELECT Visitor FROM table WHERE Home = 'los angeles'`

## Example 5 - CORRECT  (dev line 6365)
- **Question:** What away team scored 9.18 (72)?
- **Gold SQL:** `SELECT Away team FROM table WHERE Away team score = '9.18 (72)'`
- **Our SQL:** `SELECT Away team FROM table WHERE Away team score = '9.18 (72)'`

## Example 6 - WRONG  (dev line 5706)
- **Question:** What is Fitzroy's Home team Crowd?
- **Gold SQL:** `SELECT SUM(Crowd) FROM table WHERE Home team = 'fitzroy'`
- **Our SQL:** `SELECT Crowd FROM table WHERE Home team = 'fitzroy'`
- **Failure:** wrong aggregation

## Example 7 - WRONG  (dev line 4601)
- **Question:** What is the time/retired for the rider with the manufacturuer yamaha, grod of 1 and 21 total laps?
- **Gold SQL:** `SELECT Time/Retired FROM table WHERE Laps = '21' AND Manufacturer = 'yamaha' AND Grid = '1'`
- **Our SQL:** `SELECT Time/Retired FROM table WHERE Manufacturer = 'yamaha, grod of 1' AND Laps = '21'`
- **Failure:** missing condition

## Example 8 - WRONG  (dev line 3341)
- **Question:** What year was the building completed that has 10 floors?
- **Gold SQL:** `SELECT AVG(Year completed) FROM table WHERE Floors = '10'`
- **Our SQL:** `SELECT Year completed FROM table WHERE Floors = '10'`
- **Failure:** wrong aggregation

## Example 9 - WRONG  (dev line 5569)
- **Question:** Name the frequence MHz for ERP W of 55
- **Gold SQL:** `SELECT Frequency MHz FROM table WHERE ERP W = '55'`
- **Our SQL:** `SELECT FCC info FROM table WHERE ERP W = '55'`
- **Failure:** wrong select column

## Example 10 - WRONG  (dev line 4020)
- **Question:** Which Against has a Drawn smaller than 5, and a Lost smaller than 6, and a Points larger than 36?
- **Gold SQL:** `SELECT COUNT(Against) FROM table WHERE Drawn < 5 AND Lost < 6 AND Points > 36`
- **Our SQL:** `SELECT AVG(Against) FROM table WHERE Drawn < 5 AND Lost < 6 AND Points > 36`
- **Failure:** wrong aggregation
