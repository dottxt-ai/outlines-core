use anyhow::{anyhow, Result};
use std::num::NonZeroU64;

pub fn validate_quantifiers(
    min_bound: Option<u64>,
    max_bound: Option<u64>,
    start_offset: u64,
) -> Result<(Option<NonZeroU64>, Option<NonZeroU64>)> {
    let min_bound = min_bound.map(|n| NonZeroU64::new(n.saturating_sub(start_offset)));
    let max_bound = max_bound.map(|n| NonZeroU64::new(n.saturating_sub(start_offset)));

    if let (Some(min), Some(max)) = (min_bound, max_bound) {
        if max < min {
            return Err(anyhow!(
                "max bound must be greater than or equal to min bound"
            ));
        }
    }

    Ok((min_bound.flatten(), max_bound.flatten()))
}

pub fn get_num_items_pattern(min_items: Option<u64>, max_items: Option<u64>) -> Option<String> {
    let min_items = min_items.unwrap_or(0);

    match max_items {
        None => Some(format!("{{{},}}", min_items.saturating_sub(1))),
        Some(max_items) => {
            if max_items < 1 {
                None
            } else {
                Some(format!(
                    "{{{},{}}}",
                    min_items.saturating_sub(1),
                    max_items.saturating_sub(1)
                ))
            }
        }
    }
}
