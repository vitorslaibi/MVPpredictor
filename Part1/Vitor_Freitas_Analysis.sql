-- Mystics Data Analyst Position – Technical Exercise Part 1: SQL Assessment
-- Vitor Freitas

-- 1
SELECT COUNT(*) AS robert_count 
FROM players 
WHERE first_name = 'Robert';

-- 2
SELECT player_id, AVG(min) AS avg_minutes
FROM season_totals_allstar 
GROUP BY player_id 
HAVING COUNT(*) > 1;

-- 3
SELECT p.full_name
FROM players p 
LEFT JOIN career_totals_allstar a ON p.id = a.player_id 
INNER JOIN career_totals_regular_season r ON p.id = r.player_id 
WHERE a.player_id IS NULL 
ORDER BY r.min DESC;

-- 4. Output is f.Mary,2 and h.Brenda,1
-- Create the 'users' table
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    sex CHAR(1)
);

-- Insert data into the 'users' table
INSERT INTO users (id, name, sex) VALUES
(1, 'Ann', NULL),
(2, 'Steve', 'm'),
(3, 'Mary', 'f'),
(4, 'Brenda', 'f');

-- Create the 'friends' table
CREATE TABLE friends (
    user1 INT,
    user2 INT,
    FOREIGN KEY (user1) REFERENCES users(id),
    FOREIGN KEY (user2) REFERENCES users(id)
);

-- Insert data into the 'friends' table
INSERT INTO friends (user1, user2) VALUES
(1, 2),
(1, 3),
(2, 3);
