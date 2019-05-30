<template>
  <div
    ref="logCon"
    class="m-log">
    <pre class="log-container">
      <div
        v-for="(text, index) in info"
        :key="index"
        class="log-line"
      ><span class="log-num">{{ index + 1 }}</span><span :class="{'log-text': true, 'debug-log': text.indexOf('DEBUG') === 0, 'warning-log': text.indexOf('WARNING') === 0, 'error-log': text.indexOf('ERROR') === 0}">{{ text }}</span></div>
    </pre>
  </div>
</template>

<script>
import io from 'socket.io-client'

export default {
  props: {
    rows: {
      type: Array,
      default: () => []
    },
    // socket地址
    attach: {
      type: String,
      default: () => ''
    }
  },
  data () {
    return {
      info: this.rows,
      socket: null,
      stopScroll: false
    }
  },
  watch: {
    rows (val) {
      this.info = this.info.concat(val)
    },
    info () {
      this.$nextTick(() => {
        if (!this.stopScroll) {
          this.$refs['logCon'].scrollTop = this.$refs['logCon'].scrollHeight
        }
      })
    }
  },
  mounted () {
    if (this.attach) {
      this.socket = io(this.attach)

      this.socket.on('connect', () => {
        console.log('client connected')
      })

      this.socket.on('new_server_log', (data) => {
        data.split('\n').forEach((item) => {
          if (item) {
            this.info.push(item)
          }
        })
      })

      this.socket.on('disconnect', () => {
        console.log('client disconnected')
      })
    }
    this.$refs['logCon'].addEventListener('scroll', this.scrollHandler)
  },
  methods: {
    scrollHandler (e) {
      if (e.target.scrollHeight === (e.target.scrollTop + e.target.offsetHeight)) {
        this.stopScroll = false
      } else {
        this.stopScroll = true
      }
    }
  }
}
</script>

<style lang="scss">
.m-log {
  position: relative;
  min-height: 200px;
  max-height: 400px;
  padding: 10px 0;
  background-color: #222;
  overflow: auto;
  .log-fix-btn {
    position: sticky;
    top: 0;
    padding-right: 10px;
    text-align: right;
    z-index: 1;
    &:active, &:focus, &:hover {
      color: #909399;
      background: #f4f4f5;
      border-color: #d3d4d6;
    }
  }
  .log-container {
    margin: 0;
    font-size: 12px;
    color: #f1f1f1;
    line-height: 19px;
    white-space: pre-wrap;
    word-wrap: break-word;
    .log-line {
      position: relative;
      min-height: 19px;
      padding-left: 45px;
      overflow: hidden;
      .log-num {
        display: inline-block;
        text-align: right;
        min-width: 40px;
        margin-left: -40px;
        cursor: pointer;
        text-decoration: none;
        color: #666;
        &:after {
          content: '';
          padding-right: 1em;
        }
      }
      &:nth-child(even) {
        background-color: #2b2b2b;
      }
    }
    .debug-log {
      color: #B5DCFE;
    }
    .warning-log {
      color: #FFFF91;
    }
    .error-log {
      color: #FF9B93;
    }
  }
}
</style>
