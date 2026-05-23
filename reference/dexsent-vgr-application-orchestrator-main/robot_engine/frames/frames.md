# frames

Frame graph and versioned transform storage.

All transforms use explicit `parent_frame` and `child_frame` labels. Missing
frames and invalid chains raise structured/clear errors instead of silently
guessing conventions.
