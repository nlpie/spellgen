# This SQL produces note and patient count data for the base and expansion queries.
# Assign @term on the first line and run both statements
# Executed on MySQL 5.5.36

SELECT @term := 'serotonin';
SELECT @term as term, dc.id as distinct_count_id, dc.registration_id, dc.terms, dc.label,
(select hits from distinct_count where terms=@term and label=dc.label) as correct_spelling_count,
(select hits from distinct_count where terms like concat('% NOT ',@term) and label=dc.label) as potential_marginal_count,
(select hits from distinct_count where terms like concat('% NOT ',@term) and label=dc.label) as actual_marginal_count
FROM notes_next.distinct_count dc
WHERE dc.label IN ('Note Id')
AND terms LIKE concat('% NOT ',@term)
UNION ALL
SELECT @term, dc.id as distinct_count_id, dc.registration_id, dc.terms, dc.label, 
(select bucket_count from distinct_count where terms=@term and label=dc.label) as correct_spelling_count,
(select count(*) from bucket where distinct_count_id=dc.id) as potential_marginal_count, 
(select count(*) from bucket where distinct_count_id=dc.id 
	and `key` not in (
		select `key` from bucket where distinct_count_id=(
			select id from distinct_count where terms=@term and label=dc.label
		)
	)
) as actual_marginal_count
FROM notes_next.distinct_count dc
WHERE dc.label IN ('Mrn')
AND terms LIKE concat('% NOT ',@term);
