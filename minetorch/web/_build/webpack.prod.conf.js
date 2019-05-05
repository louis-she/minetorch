'use strict'

process.env.NODE_ENV = 'production';

const merge = require('webpack-merge')
const baseConfig = require('./webpack.base.conf')
const config = require('./config')
const utils = require('./utils')

module.exports = merge(baseConfig, {
  mode: process.env.NODE_ENV,
  output: {
    path: config.assetsRoot,
    filename: `./${config.assetsJsPath}[name]/build-[chunkhash:7].js`,
    chunkFilename: `./${config.assetsJsPath}[name]/build-[chunkhash:7].js`,
    publicPath: config.assetsPublicPath
  },
  optimization: {
    splitChunks: {
      cacheGroups: {
        commons: {
          test: /[\\/]node_modules[\\/]/,
          name: "common",
          chunks: "all",
        },
      },
    },
  }
})
