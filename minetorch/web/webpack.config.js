const path = require('path');
const webpack = require('webpack');

module.exports = {
  entry: './assets/application.js',
  mode: 'development',
  output: {
    path: path.resolve(__dirname, 'static'),
    filename: 'main.bundle.js'
  },
  module: {
    rules: [{
      test: /\.js$/,
      loader: 'babel-loader',
      query: {
        presets: ['@babel/preset-env']
      }
    }, {
      test: /\.scss$/,
      use: [
        "style-loader",
        "css-loader",
        "sass-loader"
      ]
    }]
  },
  stats: {
    colors: true
  },
  devtool: 'source-map'
};
