"""
$(TYPEDSIGNATURES)

Per-cell building morphometry on `grid` from a building `dataset`, as a NamedTuple of `Field`s
named after the closure inputs: `plan_area_index` and `mean_building_height` (m), and, where
the dataset measures them, `building_height_deviation` (m), `maximum_building_height` (m) and
`frontal_area_index`. An optional `measured_plan_area_index` gives the built fraction with
measured heights for neighborhood averaging. Dataset modules add methods for their products.
"""
function building_morphometry end
