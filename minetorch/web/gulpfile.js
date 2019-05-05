const gulp = require('gulp')
const zip = require('gulp-zip')
const del = require('del')
const rsync = require('gulp-rsync')

gulp.task('zip', () =>
  gulp.src('dist/**')
    .pipe(zip('dist.zip'))
    .pipe(gulp.dest('./'))
)

gulp.task('clean', () => {
  return del([
    'dist/**/*'
  ])
})
