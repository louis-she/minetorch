'use strict'

const utils = require('./utils')
require('dotenv').config({
  path: utils.resolve('../../.env')
})

const os = require('os')
const ifaces = os.networkInterfaces()
let localhost = process.env.SERVER_ADDR

module.exports = {
  devServer: {
    host: localhost,
    port: 3100,
    proxy: {
      '/api': {
        target: `http://${process.env.SERVER_ADDR}:${process.env.WEB_SERVER_PORT}`,
        changeOrigin: true
      }
    }
  },
  assetsRoot: utils.resolve('dist/'),
  assetsJsPath: 'static/js/',
  assetsCssPath: 'static/css/',
  assetsImgPath: 'static/image/',
  assetsPublicPath: process.env.BUILD_ENV === 'production' ? '' : ''
}
